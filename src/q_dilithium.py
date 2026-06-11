"""
Q-DILITHIUM — CRYSTALS-Dilithium post-quantum digital signatures
NIST FIPS 204 (2024) / CRYSTALS-Dilithium round-3 specification

Implements Dilithium2 (security level 2, λ=128 bits):
  n=256, q=8380417, k=4, l=3, η=2, τ=39, β=78, γ₁=2^17, γ₂=(q-1)/88,
  ω=80, λ=256

Conceptual counterpart of: ECDSA, RSA-PSS, Ed25519
Powers: TLS 1.3 certificate signing in the post-quantum era,
        code signing, JWT signatures, SSH host keys

Security: lattice hardness of Module-LWE + Module-SIS over Z_q[X]/(X^n+1)
References: Ducas et al. 2020, NIST FIPS 204
"""

from __future__ import annotations
import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dilithium-2 parameters (NIST FIPS 204, Table 1, security level 2)
# ---------------------------------------------------------------------------
_N    = 256        # polynomial degree
_Q    = 8_380_417  # prime modulus: 2^23 - 2^13 + 1
_K    = 4          # rows in matrix A (verification key height)
_L    = 3          # cols in matrix A (signing key width)
_ETA  = 2          # secret key range: coefficients in [-η, η]
_TAU  = 39         # number of ±1 in challenge polynomial
_BETA = 78         # β = τ·η  (bound on z = y + c·s)
_GAMMA1 = 1 << 17  # y mask: coefficients in (-γ₁, γ₁]
_GAMMA2 = (_Q - 1) // 88   # rounding range
_OMEGA  = 80       # max hint ones
_LAMBDA = 256      # commitment hash output bits

# NTT-related: primitive 256th root of unity modulo q
# ζ = 1753, ζ^256 ≡ -1 (mod q)  =>  ζ^512 ≡ 1 (mod q)
_ZETA = 1753

# Byte-lengths
_SEEDBYTES   = 32
_CRHBYTES    = 64
_PK_BYTES    = _SEEDBYTES + 32 * _K * 10 // 8 * 8   # rough; see _pack_t1
_SK_BYTES    = 3 * _SEEDBYTES + 8 * _L * 3 + 8 * _K * 3 + 8 * _K * 13
_SIG_BYTES   = _LAMBDA // 4 + _L * 32 * 18 // 8 * 8 + _OMEGA + _K


# ---------------------------------------------------------------------------
# Finite-field helpers
# ---------------------------------------------------------------------------

def _mod_q(x: int) -> int:
    return x % _Q

def _centered(x: int, q: int = _Q) -> int:
    """Reduce to (-q/2, q/2]."""
    r = x % q
    if r > q // 2:
        r -= q
    return r

def _pow_mod(base: int, exp: int, mod: int) -> int:
    return pow(base, exp, mod)


# ---------------------------------------------------------------------------
# Simple XOF / PRF using SHAKE-256 / SHA-3 (no external deps)
# ---------------------------------------------------------------------------

def _shake256(data: bytes, length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(data)
    return h.digest(length)

def _sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def _sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()

def _H(data: bytes) -> bytes:
    """H: {0,1}* → {0,1}^256  (commitment hash / challenge seed)."""
    return _shake256(data, 32)

def _G(data: bytes) -> bytes:
    """G: {0,1}* → {0,1}^512  (key expansion)."""
    return _shake256(data, 64)

def _PRF(seed: bytes, nonce: int) -> bytes:
    """PRF for sampling secret polynomials."""
    return _shake256(seed + bytes([nonce]), 64)


# ---------------------------------------------------------------------------
# NTT (Number-Theoretic Transform) over Z_q
# ---------------------------------------------------------------------------

def _br(x: int, bits: int) -> int:
    """Bit-reverse x with the given number of bits."""
    result = 0
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

# Precompute NTT twiddle factors (bit-reversed order, like ref implementation)
_ZETAS: List[int] = []
_ZETAS_INV: List[int] = []

def _precompute_zetas() -> None:
    global _ZETAS, _ZETAS_INV
    _ZETAS = [0] * 256
    _ZETAS_INV = [0] * 256
    for i in range(256):
        br = _br(i, 8)
        _ZETAS[i] = _pow_mod(_ZETA, br, _Q)
        _ZETAS_INV[i] = _pow_mod(_ZETA, _Q - 1 - br, _Q)

_precompute_zetas()

def _ntt(poly: List[int]) -> List[int]:
    """Forward NTT (in-place style, returns new list)."""
    a = list(poly)
    k = 0
    length = 128
    while length >= 1:
        start = 0
        while start < 256:
            k += 1
            zeta = _ZETAS[k]
            for j in range(start, start + length):
                t = (zeta * a[j + length]) % _Q
                a[j + length] = (a[j] - t) % _Q
                a[j] = (a[j] + t) % _Q
            start += 2 * length
        length //= 2
    return a

def _inv_ntt(poly: List[int]) -> List[int]:
    """Inverse NTT for the negacyclic NTT over Z_q[X]/(X^256+1).

    Uses -zetas[k] (negation of forward twiddle) per the Dilithium spec.
    This is NOT the same as the multiplicative inverse of zetas[k].
    """
    a = list(poly)
    k = 256
    length = 1
    while length <= 128:
        start = 0
        while start < 256:
            k -= 1
            zeta = (_Q - _ZETAS[k]) % _Q   # negacyclic: use -ζ^{br(k)}
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % _Q
                a[j + length] = (t - a[j + length]) % _Q
                a[j + length] = (zeta * a[j + length]) % _Q
            start += 2 * length
        length *= 2
    f = _pow_mod(256, _Q - 2, _Q)   # 256^{-1} mod q (1/N scaling)
    for i in range(256):
        a[i] = (a[i] * f) % _Q
    return a

def _ntt_mul(a: List[int], b: List[int]) -> List[int]:
    """Pointwise multiplication in NTT domain."""
    return [(x * y) % _Q for x, y in zip(a, b)]

def _poly_add(a: List[int], b: List[int]) -> List[int]:
    return [(x + y) % _Q for x, y in zip(a, b)]

def _poly_sub(a: List[int], b: List[int]) -> List[int]:
    return [(x - y) % _Q for x, y in zip(a, b)]


# ---------------------------------------------------------------------------
# Polynomial sampling
# ---------------------------------------------------------------------------

def _sample_uniform(seed: bytes, nonce_hi: int, nonce_lo: int) -> List[int]:
    """Uniform sampling of polynomial coefficients in [0, q)  (ExpandA)."""
    buf = _shake256(seed + bytes([nonce_lo, nonce_hi]), 840)
    poly = []
    pos = 0
    while len(poly) < _N and pos + 3 <= len(buf):
        b0, b1, b2 = buf[pos], buf[pos + 1], buf[pos + 2]
        pos += 3
        d1 = b0 | ((b1 & 0x7F) << 8)
        d2 = (b1 >> 7) | (b2 << 1)
        if d1 < _Q:
            poly.append(d1)
        if len(poly) < _N and d2 < _Q:
            poly.append(d2)
    if len(poly) < _N:
        # extend buffer if needed (rare)
        extra = _shake256(seed + bytes([nonce_lo, nonce_hi]) + b"\xff", 840)
        ei = 0
        while len(poly) < _N and ei + 3 <= len(extra):
            b0, b1, b2 = extra[ei], extra[ei + 1], extra[ei + 2]
            ei += 3
            d1 = b0 | ((b1 & 0x7F) << 8)
            if d1 < _Q and len(poly) < _N:
                poly.append(d1)
    return poly[:_N]

def _sample_eta(seed: bytes, nonce: int) -> List[int]:
    """Sample polynomial with coefficients uniformly in [-η, η]."""
    buf = _PRF(seed, nonce)
    poly = []
    for byte in buf:
        a = byte & 0x0F
        b = byte >> 4
        if a < 15:
            poly.append(2 - (a % 5))  # maps 0..4 → 2..−2 uniformly
        if len(poly) < _N and b < 15:
            poly.append(2 - (b % 5))
        if len(poly) >= _N:
            break
    # Pad if necessary (very rare)
    while len(poly) < _N:
        poly.append(0)
    return poly[:_N]

def _sample_gamma1(seed: bytes, nonce: int) -> List[int]:
    """Sample masking polynomial y with coefficients in (-γ₁, γ₁]."""
    buf = _shake256(seed + struct.pack("<H", nonce), 576)
    poly = []
    i = 0
    # γ₁ = 2^17: each coefficient needs 18 bits → 9 bytes for 4 coefficients
    while len(poly) < _N and i + 9 <= len(buf):
        chunk = buf[i:i + 9]
        i += 9
        vals = _unpack_18bit(chunk)
        for v in vals:
            poly.append(_GAMMA1 - v)
    return poly[:_N]

def _unpack_18bit(chunk: bytes) -> List[int]:
    """Unpack 4 × 18-bit integers from 9 bytes."""
    b = int.from_bytes(chunk, "little")
    mask = (1 << 18) - 1
    return [(b >> (18 * i)) & mask for i in range(4)]


def _sample_challenge(seed: bytes) -> List[int]:
    """Sample challenge polynomial c with exactly τ nonzero ±1 coefficients."""
    buf = _shake256(seed, 136)
    signs = int.from_bytes(buf[:8], "little")
    pos = 8
    c = [0] * _N
    for i in range(_N - 1, _N - _TAU - 1, -1):
        j = buf[pos] % (i + 1)
        pos += 1
        c[i] = c[j]
        c[j] = 1 - 2 * (signs & 1)
        signs >>= 1
    return c


# ---------------------------------------------------------------------------
# High/low bits, hint computation
# ---------------------------------------------------------------------------

def _power2round(r: int, d: int = 13) -> Tuple[int, int]:
    """Decompose r = r1·2^d + r0, r0 ∈ (-2^{d-1}, 2^{d-1}]."""
    r0 = r % (1 << d)
    if r0 > (1 << (d - 1)):
        r0 -= 1 << d
    r1 = (r - r0) >> d
    return r1, r0

def _decompose(r: int) -> Tuple[int, int]:
    """
    Decompose r = r1·2γ₂ + r0, r0 ∈ (-γ₂, γ₂].
    Returns (r1, r0).
    """
    r = r % _Q
    r0 = r % (2 * _GAMMA2)
    if r0 > _GAMMA2:
        r0 -= 2 * _GAMMA2
    r1 = (r - r0) // (2 * _GAMMA2)
    if r1 == (_Q - 1) // (2 * _GAMMA2):
        r1 = 0
        r0 -= 1
    return r1, r0

def _high_bits(r: int) -> int:
    return _decompose(r)[0]

def _low_bits(r: int) -> int:
    return _decompose(r)[1]

def _make_hint(z: int, r: int) -> int:
    """Return 1 if high bits of (r + z) differ from high bits of r."""
    r1 = _high_bits(r)
    v1 = _high_bits((r + z) % _Q)
    return int(r1 != v1)

def _use_hint(h: int, r: int) -> int:
    """Recover high bits using hint."""
    m = (_Q - 1) // (2 * _GAMMA2)
    r1, r0 = _decompose(r)
    if h == 1:
        if r0 > 0:
            return (r1 + 1) % m
        else:
            return (r1 - 1) % m
    return r1


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

Poly  = List[int]
Vec   = List[Poly]    # list of k or l polynomials
Mat   = List[Vec]     # k × l matrix of polynomials

def _vec_add(u: Vec, v: Vec) -> Vec:
    return [_poly_add(a, b) for a, b in zip(u, v)]

def _vec_sub(u: Vec, v: Vec) -> Vec:
    return [_poly_sub(a, b) for a, b in zip(u, v)]

def _mat_vec_mul(A: Mat, v: Vec) -> Vec:
    """Matrix-vector multiply in NTT domain."""
    k = len(A)
    l = len(v)
    result = []
    for i in range(k):
        acc = [0] * _N
        for j in range(l):
            prod = _ntt_mul(A[i][j], v[j])
            acc = _poly_add(acc, prod)
        result.append(acc)
    return result

def _ntt_vec(v: Vec) -> Vec:
    return [_ntt(p) for p in v]

def _inv_ntt_vec(v: Vec) -> Vec:
    return [_inv_ntt(p) for p in v]

def _scale_vec(c_hat: Poly, v: Vec) -> Vec:
    return [_inv_ntt(_ntt_mul(c_hat, _ntt(p))) for p in v]

def _inf_norm_vec(v: Vec) -> int:
    """∞-norm of a vector of polynomials (max |coefficient|)."""
    mx = 0
    for p in v:
        for coef in p:
            c = abs(_centered(coef))
            if c > mx:
                mx = c
    return mx


# ---------------------------------------------------------------------------
# Pack / unpack helpers (simplified for clarity — not bit-exact FIPS 204)
# ---------------------------------------------------------------------------

def _pack_poly_t1(poly: Poly) -> bytes:
    """Pack 256 10-bit integers into 320 bytes (4 coefficients per 5 bytes)."""
    out = b""
    for i in range(0, 256, 4):
        v = (poly[i] & 0x3FF) | ((poly[i+1] & 0x3FF) << 10) | \
            ((poly[i+2] & 0x3FF) << 20) | ((poly[i+3] & 0x3FF) << 30)
        out += struct.pack("<Q", v)[:5]
    return out

def _unpack_poly_t1(data: bytes) -> Poly:
    poly = []
    pos = 0
    while len(poly) < 256 and pos + 5 <= len(data):
        v = int.from_bytes(data[pos:pos + 5], "little")
        poly.extend([(v >> (10 * j)) & 0x3FF for j in range(4)])
        pos += 5
    return poly[:256]

def _pack_poly_eta(poly: Poly) -> bytes:
    """Pack 256 3-bit (η=2) integers: 3 bits each → 96 bytes."""
    out = bytearray(96)
    for i in range(128):
        a = (_ETA - poly[2 * i]) & 0x0F
        b = (_ETA - poly[2 * i + 1]) & 0x0F
        out[i] = a | (b << 4)
    return bytes(out)

def _unpack_poly_eta(data: bytes) -> Poly:
    poly = []
    for byte in data[:96]:
        poly.append(_ETA - (byte & 0x0F))
        poly.append(_ETA - (byte >> 4))
    return poly[:256]

def _pack_poly_z(poly: Poly) -> bytes:
    """Pack z coefficients in range (-γ₁, γ₁] as 18-bit unsigned offsets.

    Encoding: unsigned_val = γ₁ - coef  (coef centred in (-γ₁, γ₁])
    Four coefficients packed into 9 bytes (4 × 18 = 72 bits).
    Coefficients are first centred from [0, Q) to (-γ₁, γ₁] via _centered().
    """
    out = b""
    mask18 = (1 << 18) - 1
    for i in range(0, 256, 4):
        vals = [(_GAMMA1 - _centered(poly[i + j])) & mask18 for j in range(4)]
        v = vals[0] | (vals[1] << 18) | (vals[2] << 36) | (vals[3] << 54)
        out += v.to_bytes(9, "little")
    return out

def _unpack_poly_z(data: bytes) -> Poly:
    """Unpack z polynomial packed by _pack_poly_z.  Returns centred coefficients."""
    poly = []
    mask18 = (1 << 18) - 1
    for i in range(0, 256 * 9 // 4, 9):
        v = int.from_bytes(data[i:i + 9], "little")
        for j in range(4):
            poly.append(_GAMMA1 - ((v >> (18 * j)) & mask18))
    return poly[:256]


# ---------------------------------------------------------------------------
# Key serialisation
# ---------------------------------------------------------------------------

def _serialize_pk(rho: bytes, t1: Vec) -> bytes:
    parts = [rho]
    for p in t1:
        parts.append(_pack_poly_t1(p))
    return b"".join(parts)

def _deserialize_pk(data: bytes) -> Tuple[bytes, Vec]:
    rho = data[:32]
    t1 = []
    pos = 32
    for _ in range(_K):
        t1.append(_unpack_poly_t1(data[pos:pos + 320]))
        pos += 320
    return rho, t1

def _serialize_sk(rho: bytes, K: bytes, tr: bytes, s1: Vec, s2: Vec, t0: Vec) -> bytes:
    parts = [rho, K, tr]
    for p in s1:
        parts.append(_pack_poly_eta(p))
    for p in s2:
        parts.append(_pack_poly_eta(p))
    for p in t0:
        # Pack t0: coefficients in (-2^12, 2^12] → 13 bits each
        parts.append(struct.pack("<" + "i" * 256, *[c & 0xFFFF for c in p]))
    return b"".join(parts)

def _deserialize_sk(data: bytes) -> Tuple[bytes, bytes, bytes, Vec, Vec, Vec]:
    rho = data[:32]
    K = data[32:64]
    tr = data[64:96]
    pos = 96
    s1 = []
    for _ in range(_L):
        s1.append(_unpack_poly_eta(data[pos:pos + 96]))
        pos += 96
    s2 = []
    for _ in range(_K):
        s2.append(_unpack_poly_eta(data[pos:pos + 96]))
        pos += 96
    t0 = []
    for _ in range(_K):
        raw = struct.unpack("<" + "i" * 256, data[pos:pos + 1024])
        t0.append(list(raw))
        pos += 1024
    return rho, K, tr, s1, s2, t0


# ---------------------------------------------------------------------------
# Core Dilithium operations
# ---------------------------------------------------------------------------

def _expand_A(rho: bytes) -> Mat:
    """ExpandA: derive matrix A from seed ρ."""
    A = []
    for i in range(_K):
        row = []
        for j in range(_L):
            row.append(_sample_uniform(rho, i, j))
        A.append(row)
    return A


@dataclass
class DilithiumPublicKey:
    rho: bytes           # 32-byte matrix seed
    t1: Vec              # packed high bits of t = A·s1 + s2
    _pk_bytes: bytes = field(default=b"", repr=False)

    def to_bytes(self) -> bytes:
        if not self._pk_bytes:
            object.__setattr__(self, "_pk_bytes", _serialize_pk(self.rho, self.t1))
        return self._pk_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "DilithiumPublicKey":
        rho, t1 = _deserialize_pk(data)
        return cls(rho=rho, t1=t1, _pk_bytes=data)


@dataclass
class DilithiumPrivateKey:
    rho: bytes
    K: bytes             # 32-byte signing nonce seed
    tr: bytes            # 64-byte public-key hash
    s1: Vec              # secret l-vector (signing key)
    s2: Vec              # error k-vector
    t0: Vec              # low bits of t (for hint computation)
    public_key: DilithiumPublicKey = field(repr=False)

    def to_bytes(self) -> bytes:
        return _serialize_sk(self.rho, self.K, self.tr, self.s1, self.s2, self.t0)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DilithiumPrivateKey":
        rho, K, tr, s1, s2, t0 = _deserialize_sk(data)
        # Reconstruct public key from rho and t0/s2 (need full keygen for t1)
        # For simplicity, store pk_bytes separately; in practice embed in sk.
        raise NotImplementedError("Deserialize via keygen; SK embedding not yet implemented.")

    def sign(self, message: bytes, randomized: bool = False) -> "DilithiumSignature":
        return sign(self, message, randomized=randomized)


@dataclass
class DilithiumSignature:
    c_tilde: bytes       # 32-byte challenge hash
    z: Vec               # response vector (l polynomials)
    h: List[List[int]]   # hint bits (k polynomials, each ∈ {0,1})

    def to_bytes(self) -> bytes:
        parts = [self.c_tilde]
        for p in self.z:
            parts.append(_pack_poly_z(p))
        # Pack hints: for each of k polynomials, run-length encode positions of 1s
        hint_bytes = bytearray(_OMEGA + _K)
        idx = 0
        for i, poly in enumerate(self.h):
            ones = [j for j, v in enumerate(poly) if v]
            for pos in ones:
                hint_bytes[idx] = pos
                idx += 1
            hint_bytes[_OMEGA + i] = idx
        parts.append(bytes(hint_bytes))
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DilithiumSignature":
        c_tilde = data[:32]
        pos = 32
        z = []
        for _ in range(_L):
            z.append(_unpack_poly_z(data[pos:pos + 576]))
            pos += 576
        # Unpack hints
        hint_data = data[pos:pos + _OMEGA + _K]
        h = [[0] * _N for _ in range(_K)]
        prev = 0
        for i in range(_K):
            end = hint_data[_OMEGA + i]
            for j in range(prev, end):
                h[i][hint_data[j]] = 1
            prev = end
        return cls(c_tilde=c_tilde, z=z, h=h)


# ---------------------------------------------------------------------------
# Keygen
# ---------------------------------------------------------------------------

def keygen(seed: Optional[bytes] = None) -> Tuple[DilithiumPublicKey, DilithiumPrivateKey]:
    """
    Dilithium.KeyGen() — FIPS 204 Algorithm 6.

    Returns (pk, sk).
    """
    if seed is None:
        seed = os.urandom(_SEEDBYTES)

    xi = _G(seed)
    rho  = xi[:32]       # matrix seed (public)
    rho2 = xi[32:96]     # secret expansion seed
    K    = xi[32:64]     # signing nonce seed (subset of rho2 for simplicity)

    # ExpandA: matrix A ∈ R_q^{k×l} (in NTT domain)
    A_hat = _expand_A(rho)
    A_hat_ntt = [[_ntt(A_hat[i][j]) for j in range(_L)] for i in range(_K)]

    # Sample secret vectors s1 ∈ [-η,η]^l, s2 ∈ [-η,η]^k
    s1 = [_sample_eta(rho2, j)       for j in range(_L)]
    s2 = [_sample_eta(rho2, _L + i)  for i in range(_K)]

    # t = NTT⁻¹(A_hat · NTT(s1)) + s2
    s1_hat = [_ntt(p) for p in s1]
    As1_hat = []
    for i in range(_K):
        acc = [0] * _N
        for j in range(_L):
            acc = _poly_add(acc, _ntt_mul(A_hat_ntt[i][j], s1_hat[j]))
        As1_hat.append(acc)
    As1 = [_inv_ntt(p) for p in As1_hat]
    t = [_poly_add(As1[i], s2[i]) for i in range(_K)]

    # Power2Round: split t into (t1, t0)
    t1 = [[_power2round(c)[0] for c in p] for p in t]
    t0 = [[_power2round(c)[1] for c in p] for p in t]

    pk = DilithiumPublicKey(rho=rho, t1=t1)
    tr = _H(pk.to_bytes())   # 32-byte hash of public key (used in signing)

    sk = DilithiumPrivateKey(
        rho=rho, K=K, tr=tr,
        s1=s1, s2=s2, t0=t0,
        public_key=pk,
    )
    return pk, sk


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------

def sign(sk: DilithiumPrivateKey, message: bytes,
         randomized: bool = False) -> DilithiumSignature:
    """
    Dilithium.Sign() — FIPS 204 Algorithm 7.

    Rejection-sampling loop: sample y, compute commitment w, derive challenge c,
    compute response z = y + c·s1; reject if z or low-bits of w − c·s2 are large.
    """
    # Pre-compute NTT representations of A, s1, s2, t0
    A_hat_ntt = [[_ntt(p) for p in row] for row in _expand_A(sk.rho)]
    s1_hat = [_ntt(p) for p in sk.s1]
    s2_hat = [_ntt(p) for p in sk.s2]
    t0_hat = [_ntt(p) for p in sk.t0]

    # μ = H(tr || M)
    mu = _shake256(sk.tr + message, 64)

    rho_prime = _shake256(sk.K + (os.urandom(32) if randomized else b"\x00" * 32) + mu, 64)

    nonce = 0
    while True:
        # Sample masking vector y ∈ (-γ₁, γ₁]^l
        y = [_sample_gamma1(rho_prime, nonce + j) for j in range(_L)]
        nonce += _L

        # w = A·y
        y_hat = [_ntt(p) for p in y]
        Ay_hat = []
        for i in range(_K):
            acc = [0] * _N
            for j in range(_L):
                acc = _poly_add(acc, _ntt_mul(A_hat_ntt[i][j], y_hat[j]))
            Ay_hat.append(acc)
        w = [_inv_ntt(p) for p in Ay_hat]

        # w1 = HighBits(w)
        w1 = [[_high_bits(c % _Q) for c in p] for p in w]

        # c = SampleInBall(H(μ, w1))
        w1_packed = b"".join(struct.pack("<" + "H" * _N, *p) for p in w1)
        c_tilde = _H(mu + w1_packed)
        c_poly = _sample_challenge(c_tilde)
        c_hat = _ntt(c_poly)

        # z = y + c·s1
        cs1 = [_inv_ntt(_ntt_mul(c_hat, p)) for p in s1_hat]
        z = [_poly_add(y[j], cs1[j]) for j in range(_L)]

        # cs2 = c·s2
        cs2 = [_inv_ntt(_ntt_mul(c_hat, p)) for p in s2_hat]

        # Check: ‖z‖∞ < γ₁ - β  and  ‖LowBits(w - cs2)‖∞ < γ₂ - β
        z_norm = _inf_norm_vec(z)
        if z_norm >= _GAMMA1 - _BETA:
            continue

        r0 = [
            [_low_bits((_poly_sub([w[i][j]], [cs2[i][j]])[0]) % _Q) for j in range(_N)]
            for i in range(_K)
        ]
        r0_norm = _inf_norm_vec(r0)
        if r0_norm >= _GAMMA2 - _BETA:
            continue

        # ct0 = c·t0
        ct0 = [_inv_ntt(_ntt_mul(c_hat, p)) for p in t0_hat]

        # Compute hints h
        h = []
        hint_count = 0
        for i in range(_K):
            hi = []
            wcs2 = [(_poly_sub([w[i][j]], [cs2[i][j]])[0]) % _Q for j in range(_N)]
            for j in range(_N):
                bit = _make_hint(-ct0[i][j], (wcs2[j] + ct0[i][j]) % _Q)
                hi.append(bit)
                hint_count += bit
            h.append(hi)

        if hint_count > _OMEGA:
            continue

        # Check ct0 norm
        if _inf_norm_vec(ct0) >= _GAMMA2:
            continue

        return DilithiumSignature(c_tilde=c_tilde, z=z, h=h)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(pk: DilithiumPublicKey, message: bytes,
           sig: DilithiumSignature) -> bool:
    """
    Dilithium.Verify() — FIPS 204 Algorithm 8.

    Stateless, no secret material required.
    """
    # Check norms
    z_norm = _inf_norm_vec(sig.z)
    if z_norm >= _GAMMA1 - _BETA:
        return False
    hint_count = sum(sum(row) for row in sig.h)
    if hint_count > _OMEGA:
        return False

    A_hat_ntt = [[_ntt(p) for p in row] for row in _expand_A(pk.rho)]

    # μ = H(H(pk) || M)
    tr = _H(pk.to_bytes())
    mu = _shake256(tr + message, 64)

    c_poly = _sample_challenge(sig.c_tilde)
    c_hat = _ntt(c_poly)

    # w' = A·z − c·t1·2^d
    z_hat = [_ntt(p) for p in sig.z]
    Az_hat = []
    for i in range(_K):
        acc = [0] * _N
        for j in range(_L):
            acc = _poly_add(acc, _ntt_mul(A_hat_ntt[i][j], z_hat[j]))
        Az_hat.append(acc)

    t1_scaled = [[(c << 13) % _Q for c in p] for p in pk.t1]
    t1s_hat = [_ntt(p) for p in t1_scaled]
    ct1s = [_inv_ntt(_ntt_mul(c_hat, p)) for p in t1s_hat]

    w_prime = [_inv_ntt(_poly_sub(Az_hat[i], _ntt(ct1s[i]))) for i in range(_K)]

    # w1' = UseHint(h, w')
    w1_prime = [
        [_use_hint(sig.h[i][j], w_prime[i][j] % _Q) for j in range(_N)]
        for i in range(_K)
    ]

    # Recompute c_tilde' = H(μ, w1') and check
    w1_packed = b"".join(struct.pack("<" + "H" * _N, *p) for p in w1_prime)
    c_tilde_prime = _H(mu + w1_packed)

    return hmac.compare_digest(sig.c_tilde, c_tilde_prime)


# ---------------------------------------------------------------------------
# High-level convenience API
# ---------------------------------------------------------------------------

def generate_keypair(seed: Optional[bytes] = None) -> Tuple[DilithiumPublicKey, DilithiumPrivateKey]:
    """Alias for keygen()."""
    return keygen(seed)

def sign_message(sk: DilithiumPrivateKey, message: bytes) -> bytes:
    """Sign a message and return serialized signature bytes."""
    return sign(sk, message).to_bytes()

def verify_signature(pk: DilithiumPublicKey, message: bytes, sig_bytes: bytes) -> bool:
    """Verify a serialized signature."""
    try:
        sig = DilithiumSignature.from_bytes(sig_bytes)
        return verify(pk, message, sig)
    except Exception:
        return False

def verify_correctness(n_trials: int = 5) -> Tuple[int, int]:
    """Smoke-test: sign and verify n_trials messages. Returns (success, total)."""
    success = 0
    for i in range(n_trials):
        pk, sk = keygen()
        msg = f"test message {i}".encode()
        sig = sign(sk, msg)
        if verify(pk, msg, sig):
            success += 1
        # Tampered message must fail
        assert not verify(pk, msg + b"x", sig), "tampered message accepted!"
        # Wrong key must fail
        pk2, _ = keygen()
        assert not verify(pk2, msg, sig), "wrong key accepted!"
    return success, n_trials
