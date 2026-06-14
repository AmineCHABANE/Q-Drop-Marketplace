"""
Q-ZKP — Zero-Knowledge Proofs
  • Schnorr identification protocol     (Schnorr, CRYPTO 1989)
  • Fiat-Shamir non-interactive heuristic (Fiat & Shamir, CRYPTO 1986)
  • Pedersen commitments                (Pedersen, CRYPTO 1991)
  • Chaum-Pedersen equality-of-discrete-logs proof

A zero-knowledge proof lets a prover convince a verifier that a statement is
true while revealing NOTHING beyond its truth. The backbone of:
  - privacy coins (Zcash), zk-rollups (StarkNet, zkSync)
  - anonymous credentials, passwordless auth, e-voting
  - "prove you're over 18 without revealing your birthday"

Worked here over the 2048-bit MODP group from RFC 3526 (Group 14), a safe
prime p = 2q + 1; the order-q subgroup of quadratic residues is used so that
discrete log is hard and the group has known prime order.

EDUCATIONAL implementation — constant-time/side-channel hardening omitted.
Zero dependencies.
"""

from __future__ import annotations
import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Group parameters — RFC 3526 Group 14 (2048-bit MODP), a safe prime
# ---------------------------------------------------------------------------

_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B"
    "302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9"
    "A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6"
    "49286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8"
    "FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C"
    "180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFF"
    "FFFFFFFF"
)
P = int(_P_HEX, 16)              # 2048-bit safe prime
Q = (P - 1) // 2                 # prime order of the QR subgroup
# Generator of the order-q subgroup: square of 2 is a quadratic residue.
G = pow(2, 2, P)


def _rand_exponent() -> int:
    """A uniform secret exponent in [1, q-1]."""
    return 1 + secrets.randbelow(Q - 1)


def _hash_to_challenge(*values: int) -> int:
    """Fiat-Shamir hash: map a transcript to a challenge in [0, q)."""
    h = hashlib.sha256()
    for v in values:
        h.update(v.to_bytes((v.bit_length() + 7) // 8 or 1, "big"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") % Q


# ---------------------------------------------------------------------------
# Schnorr identification — interactive Σ-protocol
# ---------------------------------------------------------------------------

@dataclass
class SchnorrProof:
    commitment: int   # t = g^r
    challenge: int    # c
    response: int     # s = r + c·x  (mod q)


class SchnorrIdentification:
    """
    Prove knowledge of a secret x such that y = g^x mod p, revealing nothing
    about x. The classic three-move Σ-protocol: commit → challenge → respond.
    """

    def __init__(self, secret: Optional[int] = None):
        self.x = secret if secret is not None else _rand_exponent()
        self.y = pow(G, self.x, P)   # public key

    # --- Prover side ---

    def commit(self) -> Tuple[int, int]:
        """Round 1: pick nonce r, send commitment t = g^r. Returns (r, t)."""
        r = _rand_exponent()
        t = pow(G, r, P)
        return r, t

    def respond(self, r: int, challenge: int) -> int:
        """Round 3: s = r + c·x mod q."""
        return (r + challenge * self.x) % Q

    # --- Verifier side ---

    @staticmethod
    def verify(y: int, commitment: int, challenge: int, response: int) -> bool:
        """Check g^s == t · y^c  (mod p)."""
        lhs = pow(G, response, P)
        rhs = (commitment * pow(y, challenge, P)) % P
        return lhs == rhs

    def run_interactive(self, challenge: Optional[int] = None) -> SchnorrProof:
        """Convenience: run the full protocol with a (random) verifier challenge."""
        r, t = self.commit()
        c = challenge if challenge is not None else (1 + secrets.randbelow(Q - 1))
        s = self.respond(r, c)
        return SchnorrProof(commitment=t, challenge=c, response=s)


# ---------------------------------------------------------------------------
# Fiat-Shamir — make Schnorr non-interactive (a digital signature of knowledge)
# ---------------------------------------------------------------------------

def schnorr_prove_nizk(secret: int, message: bytes = b"") -> SchnorrProof:
    """
    Non-interactive Schnorr proof of knowledge of `secret`, bound to `message`.
    The verifier's challenge is replaced by a hash of the transcript — so the
    prover cannot cheat by choosing the commitment after seeing the challenge.
    This is exactly a Schnorr signature.
    """
    y = pow(G, secret, P)
    r = _rand_exponent()
    t = pow(G, r, P)
    msg_int = int.from_bytes(message, "big") if message else 0
    c = _hash_to_challenge(G, y, t, msg_int)
    s = (r + c * secret) % Q
    return SchnorrProof(commitment=t, challenge=c, response=s)


def schnorr_verify_nizk(y: int, proof: SchnorrProof, message: bytes = b"") -> bool:
    """Verify a non-interactive Schnorr proof against public key y."""
    msg_int = int.from_bytes(message, "big") if message else 0
    # Recompute the challenge — it must match (binds commitment to challenge)
    expected_c = _hash_to_challenge(G, y, proof.commitment, msg_int)
    if expected_c != proof.challenge:
        return False
    lhs = pow(G, proof.response, P)
    rhs = (proof.commitment * pow(y, proof.challenge, P)) % P
    return lhs == rhs


# ---------------------------------------------------------------------------
# Pedersen commitments — hiding + binding + additively homomorphic
# ---------------------------------------------------------------------------

# Second generator h with unknown discrete log relative to g (nothing-up-my-sleeve:
# derive from a hash, then square into the QR subgroup).
def _derive_h() -> int:
    seed = hashlib.sha256(b"q-drop pedersen generator h").digest()
    base = int.from_bytes(seed, "big") % P
    return pow(base, 2, P)   # land in the quadratic-residue subgroup

H = _derive_h()


@dataclass
class PedersenCommitment:
    commitment: int   # C = g^m · h^r mod p
    _value: int       # message m (kept by the committer only)
    _blinding: int    # randomness r (kept by the committer only)

    def open(self) -> Tuple[int, int]:
        """Reveal (m, r) to let a verifier check the commitment."""
        return self._value, self._blinding


def pedersen_commit(value: int, blinding: Optional[int] = None) -> PedersenCommitment:
    """
    Commit to `value` so that:
      - hiding:  C reveals nothing about value (perfectly hidden by r)
      - binding: the committer cannot later open C to a different value
    """
    r = blinding if blinding is not None else _rand_exponent()
    c = (pow(G, value % Q, P) * pow(H, r, P)) % P
    return PedersenCommitment(commitment=c, _value=value % Q, _blinding=r)


def pedersen_verify(commitment: int, value: int, blinding: int) -> bool:
    """Check that `commitment` opens to (value, blinding)."""
    expected = (pow(G, value % Q, P) * pow(H, blinding, P)) % P
    return commitment == expected


def pedersen_add(c1: PedersenCommitment, c2: PedersenCommitment) -> PedersenCommitment:
    """
    Homomorphic addition: Commit(m1,r1)·Commit(m2,r2) = Commit(m1+m2, r1+r2).
    Lets one prove facts about sums (e.g. a balanced ledger) without opening.
    """
    combined = (c1.commitment * c2.commitment) % P
    return PedersenCommitment(
        commitment=combined,
        _value=(c1._value + c2._value) % Q,
        _blinding=(c1._blinding + c2._blinding) % Q,
    )


# ---------------------------------------------------------------------------
# Chaum-Pedersen — prove two public keys share the same discrete log
# ---------------------------------------------------------------------------

@dataclass
class EqualityProof:
    t1: int
    t2: int
    challenge: int
    response: int


def prove_equal_discrete_log(x: int, g1: int, g2: int) -> EqualityProof:
    """
    Non-interactively prove that y1 = g1^x and y2 = g2^x share the SAME exponent
    x, without revealing x. Used in verifiable shuffles, threshold crypto, voting.
    """
    y1, y2 = pow(g1, x, P), pow(g2, x, P)
    r = _rand_exponent()
    t1, t2 = pow(g1, r, P), pow(g2, r, P)
    c = _hash_to_challenge(g1, g2, y1, y2, t1, t2)
    s = (r + c * x) % Q
    return EqualityProof(t1=t1, t2=t2, challenge=c, response=s)


def verify_equal_discrete_log(g1: int, g2: int, y1: int, y2: int,
                              proof: EqualityProof) -> bool:
    """Verify a Chaum-Pedersen equality proof."""
    c = _hash_to_challenge(g1, g2, y1, y2, proof.t1, proof.t2)
    if c != proof.challenge:
        return False
    ok1 = pow(g1, proof.response, P) == (proof.t1 * pow(y1, c, P)) % P
    ok2 = pow(g2, proof.response, P) == (proof.t2 * pow(y2, c, P)) % P
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_zkp() -> dict:
    """End-to-end: identification, signature-of-knowledge, commitment homomorphism."""
    # 1. Schnorr interactive: prove knowledge of x in y = g^x
    schnorr = SchnorrIdentification(secret=123456789)
    proof = schnorr.run_interactive()
    interactive_ok = SchnorrIdentification.verify(
        schnorr.y, proof.commitment, proof.challenge, proof.response)

    # A forged proof from someone WITHOUT the secret must fail
    fake_y = pow(G, 987654321, P)   # different secret
    forged_ok = SchnorrIdentification.verify(
        fake_y, proof.commitment, proof.challenge, proof.response)

    # 2. Fiat-Shamir non-interactive (a signature)
    nizk = schnorr_prove_nizk(123456789, message=b"login at 2026-06-14")
    nizk_ok = schnorr_verify_nizk(schnorr.y, nizk, message=b"login at 2026-06-14")
    nizk_tampered = schnorr_verify_nizk(schnorr.y, nizk, message=b"different message")

    # 3. Pedersen homomorphism: commit 30 and 12, sum opens to 42
    c1 = pedersen_commit(30)
    c2 = pedersen_commit(12)
    c_sum = pedersen_add(c1, c2)
    m, r = c_sum.open()
    homomorphic_ok = pedersen_verify(c_sum.commitment, 42, r) and m == 42

    return {
        "schnorr_interactive_verifies": interactive_ok,
        "forged_proof_rejected": not forged_ok,
        "fiat_shamir_verifies": nizk_ok,
        "fiat_shamir_tamper_rejected": not nizk_tampered,
        "pedersen_homomorphic_30_plus_12_eq_42": homomorphic_ok,
        "group_bits": P.bit_length(),
    }
