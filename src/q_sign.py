"""
q_sign.py — Hash-based post-quantum signatures (WOTS + Merkle tree)

References:
  - Lamport, "Constructing Digital Signatures from a One Way Function",
    SRI Technical Report, 1979
  - Merkle, "A Certified Digital Signature", CRYPTO 1989
  - Buchmann et al., "XMSS — A Practical Forward Secure Signature Scheme",
    PQCrypto 2011. RFC 8391.
  - NIST FIPS 205 (SLH-DSA / SPHINCS+, 2024) — standardized hash-based sigs

Why hash-based signatures matter for the quantum era:
  Shor's algorithm breaks RSA and ECDSA. Grover's algorithm only gives a
  quadratic speedup against hash functions — so SHA-256 keeps ~128 bits of
  quantum security. Hash-based signatures rely on NOTHING except the hash
  function being one-way. They are the most conservative post-quantum
  signature choice, already standardized (FIPS 205) and used to sign
  firmware at Microsoft, Cisco, and the Linux kernel module ecosystem.

Construction implemented here (an XMSS-style scheme):
  1. WOTS (Winternitz One-Time Signature, w=16):
       - private key: 67 random 32-byte strings
       - public key:  each chained through SHA-256 15 times, then hashed together
       - signing reveals intermediate chain values — each key signs ONCE
  2. Merkle tree over 2^h WOTS public keys:
       - the root is the long-lived public key
       - each signature includes the leaf index + authentication path
       - one keypair signs up to 2^h messages

This module powers the Q-Drop license system (tools/license_manager.py):
license keys are WOTS-signed payloads verifiable offline against the
published Merkle root — no server, no database, quantum-safe.

Pure Python 3, zero dependencies.
"""

import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


# WOTS parameters (w = 16 → nibble-based chains)
_N = 32                  # hash output bytes (SHA-256)
_W = 16                  # Winternitz parameter
_LEN1 = 64               # 32-byte digest → 64 nibbles
_LEN2 = 3                # checksum: max sum 64·15 = 960 < 16³
_LEN = _LEN1 + _LEN2     # 67 chains total
_CHAIN_MAX = _W - 1      # 15 chain steps


def _H(*parts: bytes) -> bytes:
    """Domain-separated SHA-256."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def _prf(seed: bytes, *indices: int) -> bytes:
    """Deterministic 32-byte value from a seed + index path (key derivation)."""
    msg = b"".join(struct.pack(">I", i) for i in indices)
    return hmac.new(seed, msg, hashlib.sha256).digest()


def _chain(value: bytes, start: int, steps: int, tag: bytes) -> bytes:
    """Apply the hash chain `steps` times starting from position `start`."""
    out = value
    for i in range(start, start + steps):
        out = _H(b"chain", tag, struct.pack(">I", i), out)
    return out


def _message_nibbles(digest: bytes) -> List[int]:
    """Digest → 64 message nibbles + 3 checksum nibbles (in [0, 15])."""
    nibbles = []
    for byte in digest:
        nibbles.append(byte >> 4)
        nibbles.append(byte & 0xF)
    # Checksum prevents forgery by chain extension: lowering any message
    # nibble RAISES the checksum, which the forger cannot un-hash.
    checksum = sum(_CHAIN_MAX - m for m in nibbles)
    for shift in (8, 4, 0):
        nibbles.append((checksum >> shift) & 0xF)
    return nibbles


# ---------------------------------------------------------------------------
# WOTS one-time keys
# ---------------------------------------------------------------------------

def _wots_secret(master_seed: bytes, leaf: int) -> List[bytes]:
    """Derive the 67 chain-start secrets for one leaf from the master seed."""
    return [_prf(master_seed, leaf, j) for j in range(_LEN)]


def _wots_public(master_seed: bytes, leaf: int) -> bytes:
    """Public key of one WOTS leaf: hash of all fully-chained endpoints."""
    tag = struct.pack(">I", leaf)
    secrets = _wots_secret(master_seed, leaf)
    ends = [_chain(s, 0, _CHAIN_MAX, tag) for s in secrets]
    return _H(b"wotspk", tag, *ends)


def _wots_sign(master_seed: bytes, leaf: int, digest: bytes) -> List[bytes]:
    """Sign a digest: reveal each chain advanced to its message nibble."""
    tag = struct.pack(">I", leaf)
    secrets = _wots_secret(master_seed, leaf)
    nibbles = _message_nibbles(digest)
    return [_chain(s, 0, m, tag) for s, m in zip(secrets, nibbles)]


def _wots_pk_from_sig(leaf: int, digest: bytes, sig: List[bytes]) -> bytes:
    """Complete each chain from the revealed value → candidate public key."""
    tag = struct.pack(">I", leaf)
    nibbles = _message_nibbles(digest)
    ends = [_chain(s, m, _CHAIN_MAX - m, tag) for s, m in zip(sig, nibbles)]
    return _H(b"wotspk", tag, *ends)


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------

def _merkle_parent(left: bytes, right: bytes) -> bytes:
    return _H(b"node", left, right)


def _build_tree(leaves: List[bytes]) -> List[List[bytes]]:
    """Return all tree levels, levels[0] = leaves, levels[-1] = [root]."""
    levels = [leaves]
    cur = leaves
    while len(cur) > 1:
        cur = [_merkle_parent(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)]
        levels.append(cur)
    return levels


@dataclass
class Signature:
    """A one-time signature + its position and Merkle authentication path."""
    leaf_index: int
    wots_sig: List[bytes]        # 67 × 32 bytes
    auth_path: List[bytes]       # h × 32 bytes

    def to_bytes(self) -> bytes:
        out = struct.pack(">IH", self.leaf_index, len(self.auth_path))
        for s in self.wots_sig:
            out += s
        for a in self.auth_path:
            out += a
        return out

    @classmethod
    def from_bytes(cls, blob: bytes) -> "Signature":
        leaf_index, h = struct.unpack(">IH", blob[:6])
        off = 6
        wots_sig = [blob[off + i * _N: off + (i + 1) * _N] for i in range(_LEN)]
        off += _LEN * _N
        auth_path = [blob[off + i * _N: off + (i + 1) * _N] for i in range(h)]
        expected = 6 + (_LEN + h) * _N
        if len(blob) != expected:
            raise ValueError(f"Bad signature length {len(blob)}, expected {expected}")
        return cls(leaf_index, wots_sig, auth_path)


class MerkleSigner:
    """
    Stateful hash-based signer. One instance = one keypair = 2^height signatures.

    Usage:
        signer = MerkleSigner.generate(height=10, seed=os.urandom(32))
        root   = signer.public_root          # publish this
        sig    = signer.sign(b"message")     # consumes one leaf
        assert verify(b"message", sig, root)

    CRITICAL: each leaf signs exactly once. Reusing a leaf leaks enough chain
    values for forgery — the signer tracks `next_index` and refuses reuse.
    State must be persisted between signs (see to_state / from_state).
    """

    def __init__(self, master_seed: bytes, height: int, next_index: int = 0):
        if not (1 <= height <= 20):
            raise ValueError("height must be in [1, 20]")
        if len(master_seed) < 16:
            raise ValueError("master_seed must be at least 16 bytes")
        self.master_seed = master_seed
        self.height = height
        self.n_leaves = 1 << height
        self.next_index = next_index
        leaves = [_wots_public(master_seed, i) for i in range(self.n_leaves)]
        self._levels = _build_tree(leaves)

    @classmethod
    def generate(cls, height: int = 10, seed: Optional[bytes] = None) -> "MerkleSigner":
        return cls(seed if seed is not None else os.urandom(32), height)

    @property
    def public_root(self) -> bytes:
        return self._levels[-1][0]

    @property
    def signatures_remaining(self) -> int:
        return self.n_leaves - self.next_index

    def sign(self, message: bytes) -> Signature:
        if self.next_index >= self.n_leaves:
            raise RuntimeError("Keypair exhausted — all one-time leaves used")
        leaf = self.next_index
        self.next_index += 1          # consume BEFORE signing: never reuse

        digest = _H(b"msg", message)
        wots_sig = _wots_sign(self.master_seed, leaf, digest)

        auth_path = []
        idx = leaf
        for level in self._levels[:-1]:
            sibling = idx ^ 1
            auth_path.append(level[sibling])
            idx >>= 1
        return Signature(leaf, wots_sig, auth_path)

    # -- state persistence (the index MUST survive restarts) ----------------

    def to_state(self) -> dict:
        return {
            "master_seed": self.master_seed.hex(),
            "height": self.height,
            "next_index": self.next_index,
        }

    @classmethod
    def from_state(cls, state: dict) -> "MerkleSigner":
        return cls(
            bytes.fromhex(state["master_seed"]),
            state["height"],
            state["next_index"],
        )


def verify(message: bytes, sig: Signature, public_root: bytes) -> bool:
    """
    Verify a signature against the Merkle root. Stateless, offline,
    quantum-safe: security reduces entirely to SHA-256 preimage resistance.
    """
    digest = _H(b"msg", message)
    node = _wots_pk_from_sig(sig.leaf_index, digest, sig.wots_sig)
    idx = sig.leaf_index
    for sibling in sig.auth_path:
        if idx & 1:
            node = _merkle_parent(sibling, node)
        else:
            node = _merkle_parent(node, sibling)
        idx >>= 1
    return hmac.compare_digest(node, public_root)
