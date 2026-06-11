"""
q_kyber.py — CRYSTALS-Kyber Post-Quantum Key Encapsulation (educational)

Reference: J. Bos et al., "CRYSTALS-Kyber: a CCA-secure module-lattice-based KEM",
           IEEE EuroS&P 2018. NIST FIPS 203 (2024).
           https://pq-crystals.org/kyber/

Kyber is the post-quantum standard replacing RSA/ECC in TLS, SSH, Signal.
It is based on Module-LWE (Learning with Errors over polynomial rings).

Security foundation:
  Given (A, b = A·s + e) where A is random, s is secret, e is small noise,
  it is computationally hard to find s even with a quantum computer.
  Shor's algorithm breaks RSA/ECC but gives no advantage against LWE.

This is a PEDAGOGICAL implementation:
  - Exact algorithm structure, not side-channel hardened
  - Uses Python's built-in random (production uses a CSPRNG / SHAKE-256)
  - Polynomial arithmetic is over Z_q[X]/(X^n + 1), modular ring
  - Simplified key derivation (production uses hash-based KDF)

Parameters follow Kyber-512 (security level ≈ AES-128):
  n = 256 (polynomial degree)
  k = 2  (module rank)
  q = 3329 (prime modulus)
  η1 = 3, η2 = 2 (noise bounds)
"""

import random
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Kyber-512 parameters
# ---------------------------------------------------------------------------

N = 256          # polynomial ring degree: Z_q[X]/(X^256 + 1)
K = 2            # module rank (Kyber-512)
Q = 3329         # prime modulus
ETA1 = 3         # key-generation noise bound
ETA2 = 2         # encapsulation noise bound
DU = 10          # ciphertext compression bits for u
DV = 4           # ciphertext compression bits for v


# ---------------------------------------------------------------------------
# Polynomial ring arithmetic over Z_q[X]/(X^n + 1)
# ---------------------------------------------------------------------------

Poly = List[int]   # list of n integers mod Q


def poly_zero() -> Poly:
    return [0] * N


def poly_add(a: Poly, b: Poly) -> Poly:
    return [(x + y) % Q for x, y in zip(a, b)]


def poly_sub(a: Poly, b: Poly) -> Poly:
    return [(x - y) % Q for x, y in zip(a, b)]


def poly_ntt_schoolbook(a: Poly, b: Poly) -> Poly:
    """
    Schoolbook multiplication in Z_q[X]/(X^n + 1).
    X^n ≡ -1 mod (X^n + 1), so monomials wrap with negation.
    O(n²) — production Kyber uses NTT for O(n log n).
    """
    result = [0] * N
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            idx = i + j
            coeff = (ai * bj) % Q
            if idx < N:
                result[idx] = (result[idx] + coeff) % Q
            else:
                # X^(idx) = X^(idx - n) · X^n ≡ -X^(idx-n)
                result[idx - N] = (result[idx - N] - coeff) % Q
    return result


def poly_mul(a: Poly, b: Poly) -> Poly:
    return poly_ntt_schoolbook(a, b)


# Module (vector of k polynomials)
Module = List[Poly]
Matrix = List[List[Poly]]   # k×k matrix of polynomials


def mod_add(u: Module, v: Module) -> Module:
    return [poly_add(u[i], v[i]) for i in range(K)]


def mod_sub(u: Module, v: Module) -> Module:
    return [poly_sub(u[i], v[i]) for i in range(K)]


def mat_vec_mul(A: Matrix, s: Module) -> Module:
    """Matrix-vector product: A·s, each entry is inner product of polys."""
    result = []
    for row in A:
        acc = poly_zero()
        for aij, sj in zip(row, s):
            acc = poly_add(acc, poly_mul(aij, sj))
        result.append(acc)
    return result


def mat_transpose_vec_mul(A: Matrix, u: Module) -> Module:
    """Aᵀ·u"""
    result = []
    for j in range(K):
        acc = poly_zero()
        for i in range(K):
            acc = poly_add(acc, poly_mul(A[i][j], u[i]))
        result.append(acc)
    return result


def inner_product(u: Module, v: Module) -> Poly:
    """⟨u, v⟩ = Σ u_i · v_i"""
    acc = poly_zero()
    for ui, vi in zip(u, v):
        acc = poly_add(acc, poly_mul(ui, vi))
    return acc


# ---------------------------------------------------------------------------
# Sampling (CBD = centered binomial distribution)
# ---------------------------------------------------------------------------

def _sample_cbd(eta: int, rng: random.Random) -> Poly:
    """
    Sample a polynomial with coefficients from centered binomial distribution.
    Each coefficient = Σ(η bits) − Σ(η bits), range [−η, +η].
    This produces small-norm polynomials (the 'noise' in LWE).
    """
    coeffs = []
    for _ in range(N):
        a = sum(rng.randint(0, 1) for _ in range(eta))
        b = sum(rng.randint(0, 1) for _ in range(eta))
        coeffs.append((a - b) % Q)
    return coeffs


def _sample_uniform(rng: random.Random) -> Poly:
    """Uniform random polynomial in Z_q — public matrix entries."""
    return [rng.randint(0, Q - 1) for _ in range(N)]


def _sample_matrix(rng: random.Random) -> Matrix:
    return [[_sample_uniform(rng) for _ in range(K)] for _ in range(K)]


def _sample_small_module(eta: int, rng: random.Random) -> Module:
    return [_sample_cbd(eta, rng) for _ in range(K)]


# ---------------------------------------------------------------------------
# Compression / decompression
# ---------------------------------------------------------------------------

def _compress_coeff(x: int, d: int) -> int:
    """Round (2^d / q) · x to nearest integer, mod 2^d."""
    return round((1 << d) * x / Q) % (1 << d)


def _decompress_coeff(x: int, d: int) -> int:
    """Invert compression: (q / 2^d) · x."""
    return round(Q * x / (1 << d)) % Q


def _compress_poly(p: Poly, d: int) -> List[int]:
    return [_compress_coeff(c, d) for c in p]


def _decompress_poly(p: List[int], d: int) -> Poly:
    return [_decompress_coeff(c, d) for c in p]


def _compress_module(m: Module, d: int) -> List[List[int]]:
    return [_compress_poly(p, d) for p in m]


def _decompress_module(m: List[List[int]], d: int) -> Module:
    return [_decompress_poly(p, d) for p in m]


# ---------------------------------------------------------------------------
# Message encoding / decoding
# ---------------------------------------------------------------------------

def _encode_message(msg_bytes: bytes) -> Poly:
    """
    Encode 32 bytes (256 bits) as a polynomial.
    Bit 1 → q/2 ≈ 1665 (the 'one' representative)
    Bit 0 → 0   (the 'zero' representative)
    """
    if len(msg_bytes) != 32:
        raise ValueError("Message must be exactly 32 bytes")
    poly = []
    for byte in msg_bytes:
        for bit_pos in range(8):
            bit = (byte >> bit_pos) & 1
            poly.append((Q // 2) * bit)
    return poly


def _decode_message(poly: Poly) -> bytes:
    """
    Decode: coefficient nearest to q/2 → 1, nearest to 0 → 0.
    Threshold at q/4.
    """
    bits = []
    for c in poly:
        c = c % Q
        # Distance to 0 vs distance to q/2
        dist0 = min(c, Q - c)
        dist_half = min(abs(c - Q // 2), Q - abs(c - Q // 2))
        bits.append(1 if dist_half < dist0 else 0)
    result = bytearray(32)
    for i, bit in enumerate(bits):
        result[i // 8] |= bit << (i % 8)
    return bytes(result)


# ---------------------------------------------------------------------------
# Public-key derivation (production uses SHA3-512 / SHAKE-128)
# ---------------------------------------------------------------------------

def _derive_rng(seed: bytes) -> random.Random:
    """Deterministic RNG from seed (production: SHAKE-256 XOF)."""
    digest = hashlib.sha256(seed).digest()
    return random.Random(int.from_bytes(digest, "big"))


# ---------------------------------------------------------------------------
# Kyber KEM: KeyGen / Encap / Decap
# ---------------------------------------------------------------------------

@dataclass
class KyberPublicKey:
    A: Matrix          # k×k public matrix
    t: Module          # public vector: t = A·s + e


@dataclass
class KyberPrivateKey:
    s: Module          # secret key (small-norm)
    pk: KyberPublicKey # retain public key for re-encryption check


@dataclass
class KyberCiphertext:
    u: List[List[int]]  # compressed module (k polynomials)
    v: List[int]        # compressed polynomial


def keygen(seed: bytes = None) -> Tuple[KyberPublicKey, KyberPrivateKey]:
    """
    Kyber key generation.

    A ← uniform(Z_q^{k×k})   — public matrix (from seed)
    s ← CBD(η1)^k             — secret vector (small coefficients)
    e ← CBD(η1)^k             — key-generation noise
    t = A·s + e               — public key
    """
    if seed is None:
        seed = random.randbytes(32)

    rng_A = _derive_rng(b"matrix:" + seed)
    rng_s = _derive_rng(b"secret:" + seed)

    A = _sample_matrix(rng_A)
    s = _sample_small_module(ETA1, rng_s)
    e = _sample_small_module(ETA1, rng_s)

    t = mod_add(mat_vec_mul(A, s), e)

    pk = KyberPublicKey(A=A, t=t)
    sk = KyberPrivateKey(s=s, pk=pk)
    return pk, sk


def encapsulate(pk: KyberPublicKey, message: bytes = None) -> Tuple[KyberCiphertext, bytes]:
    """
    Kyber encapsulation: produce ciphertext + shared secret.

    r  ← CBD(η1)^k            — ephemeral random vector
    e1 ← CBD(η2)^k            — ciphertext noise (u component)
    e2 ← CBD(η2)              — ciphertext noise (v component)

    u  = Aᵀ·r + e1            — first ciphertext component
    v  = ⟨t, r⟩ + e2 + m      — second ciphertext component (embeds message)

    Shared secret = H(m)       — derived from the plaintext message
    """
    if message is None:
        message = random.randbytes(32)

    rng = _derive_rng(b"encap:" + message)

    r  = _sample_small_module(ETA1, rng)
    e1 = _sample_small_module(ETA2, rng)
    e2 = _sample_cbd(ETA2, rng)

    # u = Aᵀr + e1
    u_raw = mod_add(mat_transpose_vec_mul(pk.A, r), e1)
    # v = ⟨t, r⟩ + e2 + encode(m)
    v_raw = poly_add(
        poly_add(inner_product(pk.t, r), e2),
        _encode_message(message)
    )

    # Compress for transmission
    u_c = _compress_module(u_raw, DU)
    v_c = _compress_poly(v_raw, DV)

    ct = KyberCiphertext(u=u_c, v=v_c)
    shared_secret = hashlib.sha256(b"shared:" + message).digest()
    return ct, shared_secret


def decapsulate(sk: KyberPrivateKey, ct: KyberCiphertext) -> bytes:
    """
    Kyber decapsulation: recover shared secret from ciphertext.

    Decompress u, v, then compute:
      m' = v - ⟨s, u⟩          — noise cancels: e2 + encode(m) + ⟨e, r⟩ ≈ encode(m)
      shared_secret = H(decode(m'))
    """
    # Decompress
    u = _decompress_module(ct.u, DU)
    v = _decompress_poly(ct.v, DV)

    # m' = v - ⟨s, u⟩
    # The noise terms (e·r, e1·s, e2) are small; they don't flip message bits
    su = inner_product(sk.s, u)
    m_poly = poly_sub(v, su)

    # Recover message
    message = _decode_message(m_poly)
    shared_secret = hashlib.sha256(b"shared:" + message).digest()
    return shared_secret


# ---------------------------------------------------------------------------
# Utility: correctness check
# ---------------------------------------------------------------------------

def verify_correctness(n_trials: int = 10) -> Tuple[int, int]:
    """
    Run n_trials key exchanges and count successes.
    Returns (successes, n_trials).
    """
    successes = 0
    for i in range(n_trials):
        seed = i.to_bytes(4, "big") + b"\x00" * 28
        pk, sk = keygen(seed)
        message = (i * 7 + 13).to_bytes(4, "big") + b"\x00" * 28
        ct, ss_enc = encapsulate(pk, message)
        ss_dec = decapsulate(sk, ct)
        if ss_enc == ss_dec:
            successes += 1
    return successes, n_trials
