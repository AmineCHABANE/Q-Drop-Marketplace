"""
Q-RS — Reed-Solomon erasure coding (durability without full replication)
Reed & Solomon, "Polynomial Codes over Certain Finite Fields", 1960

Split data into k shards, compute m parity shards; ANY k of the n = k+m shards
can reconstruct the original. Lose up to m shards, lose nothing. The math behind:
  - RAID-6, ZFS RAID-Z, Backblaze Vaults
  - distributed object stores (Ceph, HDFS-EC, MinIO, S3)
  - QR codes, DVDs, deep-space telemetry (Voyager)

Implemented over the Galois field GF(2^8) with primitive polynomial 0x11D
(the same field AES uses). A systematic Vandermonde generator matrix keeps the
original data shards verbatim; decoding solves a linear system over GF(256)
via Gaussian elimination.

Zero dependencies.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# GF(2^8) arithmetic — exp/log tables for fast multiply
# ---------------------------------------------------------------------------

_PRIM = 0x11D   # primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
_EXP = [0] * 512
_LOG = [0] * 256

def _init_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _PRIM
    # Duplicate exp table so we can index up to 510 without modulo
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]

_init_tables()


def gf_add(a: int, b: int) -> int:
    """Addition in GF(2^8) is XOR (also equals subtraction)."""
    return a ^ b

def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]

def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]

def gf_pow(a: int, n: int) -> int:
    if n == 0:
        return 1          # a^0 = 1 for all a (incl. 0) — needed for Vandermonde
    if a == 0:
        return 0
    return _EXP[(_LOG[a] * n) % 255]

def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF inverse of zero")
    return _EXP[(255 - _LOG[a]) % 255]


# ---------------------------------------------------------------------------
# Matrix operations over GF(2^8)
# ---------------------------------------------------------------------------

Matrix = List[List[int]]

def _mat_mul(a: Matrix, b: Matrix) -> Matrix:
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(cols):
                out[i][j] ^= gf_mul(aik, b[k][j])
    return out

def _mat_invert(m: Matrix) -> Matrix:
    """Invert a square matrix over GF(2^8) by Gauss-Jordan elimination."""
    n = len(m)
    # Augment with identity
    aug = [row[:] + [1 if i == j else 0 for j in range(n)]
           for i, row in enumerate(m)]

    for col in range(n):
        # Find a pivot
        pivot = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("matrix is singular — cannot invert")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        # Normalize pivot row
        inv = gf_inv(aug[col][col])
        aug[col] = [gf_mul(v, inv) for v in aug[col]]

        # Eliminate this column from all other rows
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [aug[r][j] ^ gf_mul(factor, aug[col][j])
                          for j in range(2 * n)]

    return [row[n:] for row in aug]


def _vandermonde(n: int, k: int) -> Matrix:
    """An n×k Vandermonde matrix V[i][j] = i^j over GF(2^8)."""
    return [[gf_pow(i, j) for j in range(k)] for i in range(n)]


def _systematic_matrix(n: int, k: int) -> Matrix:
    """
    Build an n×k generator matrix whose top k×k block is the identity (so the
    first k shards are the original data verbatim — "systematic" encoding).

    Take a Vandermonde matrix and right-multiply by the inverse of its top
    k×k block; the result has an identity on top and stays MDS (any k rows
    remain invertible).
    """
    v = _vandermonde(n, k)
    top = [row[:] for row in v[:k]]
    top_inv = _mat_invert(top)
    return _mat_mul(v, top_inv)


# ---------------------------------------------------------------------------
# Reed-Solomon codec
# ---------------------------------------------------------------------------

class ReedSolomon:
    """
    Systematic Reed-Solomon codec over GF(2^8).

    :param k: number of data shards
    :param m: number of parity shards  (can tolerate up to m lost shards)

    n = k + m total shards, with k + m ≤ 256.
    """

    def __init__(self, k: int, m: int):
        if k < 1 or m < 1:
            raise ValueError("k and m must be ≥ 1")
        if k + m > 256:
            raise ValueError("k + m must be ≤ 256 for GF(2^8)")
        self.k = k
        self.m = m
        self.n = k + m
        self._gen = _systematic_matrix(self.n, k)   # n×k generator

    def encode(self, data: bytes) -> List[bytes]:
        """
        Encode `data` into n shards (k systematic data shards + m parity).
        Data is zero-padded so its length is a multiple of k.
        """
        # Pad so length is divisible by k; remember original length via caller
        padded = bytearray(data)
        while len(padded) % self.k != 0:
            padded.append(0)
        shard_len = len(padded) // self.k

        # data_matrix: k × shard_len  (row i = i-th data shard)
        data_matrix = [list(padded[i * shard_len:(i + 1) * shard_len])
                       for i in range(self.k)]

        # all shards = gen (n×k) · data (k×shard_len)
        coded = _mat_mul(self._gen, data_matrix)
        return [bytes(row) for row in coded]

    def decode(self, shards: List[Optional[bytes]]) -> bytes:
        """
        Reconstruct the original (padded) data from any k present shards.
        Missing shards are passed as None. Raises if fewer than k are available.
        """
        if len(shards) != self.n:
            raise ValueError(f"expected {self.n} shard slots, got {len(shards)}")

        present = [(i, s) for i, s in enumerate(shards) if s is not None]
        if len(present) < self.k:
            raise ValueError(
                f"need at least {self.k} shards, only {len(present)} present")

        # Use the first k available shards
        present = present[: self.k]
        shard_len = len(present[0][1])

        # Sub-matrix of the generator for the surviving shard indices
        sub = [self._gen[i] for i, _ in present]
        sub_inv = _mat_invert(sub)

        # Stack the surviving shards as a k × shard_len matrix
        coded_matrix = [list(s) for _, s in present]

        # original data = sub_inv · surviving_shards
        data_matrix = _mat_mul(sub_inv, coded_matrix)

        out = bytearray()
        for row in data_matrix:
            out.extend(row)
        return bytes(out)

    def __repr__(self) -> str:
        return f"ReedSolomon(k={self.k}, m={self.m}, n={self.n})"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_reed_solomon() -> dict:
    """
    Encode a message into 10 shards (6 data + 4 parity), then DELETE 4 shards
    and reconstruct the original from the remaining 6. Lose 40% of the storage,
    lose zero data — at 1.67× overhead instead of replication's 2–3×.
    """
    rs = ReedSolomon(k=6, m=4)
    message = b"Quantum-safe distributed storage durability via erasure coding."
    original_len = len(message)

    shards = rs.encode(message)

    # Simulate catastrophic loss: drop 4 of the 10 shards (indices 1,4,7,9)
    damaged: List[Optional[bytes]] = list(shards)
    for lost in (1, 4, 7, 9):
        damaged[lost] = None

    recovered = rs.decode(damaged)[:original_len]

    return {
        "k_data_shards": rs.k,
        "m_parity_shards": rs.m,
        "total_shards": rs.n,
        "storage_overhead": round(rs.n / rs.k, 2),
        "shards_lost": 4,
        "message": message.decode(),
        "recovered_ok": recovered == message,
    }
