"""
q_qsf.py — QSF: a Quantum-Safe File container format

QSF is the answer to a concrete question: what does a file format look like
when EVERY cryptographic primitive inside it must survive a quantum computer?

  Classical container (e.g. age, PGP):     QSF:
  ───────────────────────────────────      ─────────────────────────────────
  RSA / X25519 key exchange          →     CRYSTALS-Kyber KEM   (src/q_kyber)
  Ed25519 / RSA signatures           →     WOTS+Merkle hash sig (src/q_sign)
  AES stream                         →     SHA-256 counter keystream*
  DEFLATE compression                →     LZSS+Huffman          (src/q_pack)

  * Symmetric crypto only needs doubled key sizes against Grover; a SHA-256
    keystream is used here for zero-dependency clarity. Production would use
    AES-256-GCM — the construction slot is the point, not the cipher choice.

Binary layout (big-endian):

  offset  size  field
  ──────  ────  ─────────────────────────────────────────────
  0       4     magic  "QSF1"
  4       1     flags  bit0=compressed  bit1=encrypted  bit2=signed
  5       4     metadata length M
  9       M     metadata (JSON, UTF-8) — creator, timestamp, content-type...
  ...     4     section length, then section — repeated:
                  [kyber ct]   if encrypted: serialized KyberCiphertext
                  [signature]  if signed: q_sign Signature bytes
                  [payload]    the (compressed,) (encrypted,) data
  last    32    SHA-256 of everything before it (integrity tripwire)

The integrity hash detects corruption; AUTHENTICITY comes from the optional
hash-based signature, which covers the payload + metadata and verifies
offline against a published Merkle root.

Pure Python 3, zero dependencies beyond sibling modules.
"""

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import q_pack
import q_kyber
import q_sign


MAGIC = b"QSF1"

_FLAG_COMPRESSED = 1
_FLAG_ENCRYPTED = 2
_FLAG_SIGNED = 4


# ---------------------------------------------------------------------------
# Kyber ciphertext (de)serialization — coefficients are < 2^16
# ---------------------------------------------------------------------------

def _serialize_kyber_ct(ct: q_kyber.KyberCiphertext) -> bytes:
    out = bytearray(struct.pack(">HH", len(ct.u), len(ct.u[0])))
    for poly in ct.u:
        for c in poly:
            out += struct.pack(">H", c)
    out += struct.pack(">H", len(ct.v))
    for c in ct.v:
        out += struct.pack(">H", c)
    return bytes(out)


def _deserialize_kyber_ct(blob: bytes) -> q_kyber.KyberCiphertext:
    k, n = struct.unpack(">HH", blob[:4])
    off = 4
    u = []
    for _ in range(k):
        poly = list(struct.unpack(f">{n}H", blob[off:off + 2 * n]))
        off += 2 * n
        u.append(poly)
    vn, = struct.unpack(">H", blob[off:off + 2])
    off += 2
    v = list(struct.unpack(f">{vn}H", blob[off:off + 2 * vn]))
    return q_kyber.KyberCiphertext(u=u, v=v)


# ---------------------------------------------------------------------------
# Symmetric layer: SHA-256 counter keystream (XOR)
# ---------------------------------------------------------------------------

def _keystream_xor(key: bytes, data: bytes) -> bytes:
    """
    XOR data with a keystream of SHA-256(key || counter) blocks.
    Symmetric — the same call encrypts and decrypts.
    Grover only halves symmetric security: a 256-bit key keeps 128 bits.
    """
    out = bytearray(len(data))
    block = 0
    for i in range(0, len(data), 32):
        ks = hashlib.sha256(key + struct.pack(">Q", block)).digest()
        chunk = data[i:i + 32]
        for j, b in enumerate(chunk):
            out[i + j] = b ^ ks[j]
        block += 1
    return bytes(out)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _put_section(buf: bytearray, section: bytes) -> None:
    buf += struct.pack(">I", len(section))
    buf += section


def _get_section(blob: bytes, off: int) -> Tuple[bytes, int]:
    n, = struct.unpack(">I", blob[off:off + 4])
    off += 4
    return blob[off:off + n], off + n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class QSFContent:
    payload: bytes
    metadata: dict
    compressed: bool
    encrypted: bool
    signed: bool
    signature_valid: Optional[bool]   # None when unsigned or no root given


def create(
    payload: bytes,
    metadata: Optional[dict] = None,
    *,
    compress: bool = True,
    recipient_pk: Optional[q_kyber.KyberPublicKey] = None,
    signer: Optional[q_sign.MerkleSigner] = None,
    _kem_message: Optional[bytes] = None,
) -> bytes:
    """
    Build a QSF container.

    compress      — apply LZSS+Huffman before anything else
    recipient_pk  — if given, encrypt for this Kyber public key
    signer        — if given, sign payload+metadata (consumes one OTS leaf)
    _kem_message  — test hook: fixes the Kyber encapsulation randomness
    """
    metadata = dict(metadata or {})
    flags = 0
    body = payload

    if compress:
        flags |= _FLAG_COMPRESSED
        body = q_pack.compress(body)

    meta_bytes = json.dumps(metadata, sort_keys=True).encode()

    buf = bytearray(MAGIC)
    flags_pos = len(buf)
    buf.append(0)                       # flags — patched below
    buf += struct.pack(">I", len(meta_bytes))
    buf += meta_bytes

    if recipient_pk is not None:
        flags |= _FLAG_ENCRYPTED
        ct, shared = q_kyber.encapsulate(recipient_pk, _kem_message)
        body = _keystream_xor(shared, body)
        _put_section(buf, _serialize_kyber_ct(ct))

    if signer is not None:
        flags |= _FLAG_SIGNED
        # Sign what the verifier can see: metadata + final payload bytes
        sig = signer.sign(meta_bytes + body)
        _put_section(buf, sig.to_bytes())

    _put_section(buf, body)
    buf[flags_pos] = flags

    buf += hashlib.sha256(bytes(buf)).digest()
    return bytes(buf)


def open_container(
    blob: bytes,
    *,
    sk: Optional[q_kyber.KyberPrivateKey] = None,
    signer_root: Optional[bytes] = None,
) -> QSFContent:
    """
    Parse, verify and unwrap a QSF container.

    Raises ValueError on bad magic, corruption (integrity hash mismatch),
    a missing decryption key, or an INVALID signature when signer_root is
    provided. An unsigned container with signer_root given also fails —
    absence of a signature must not pass for a valid one.
    """
    if blob[:4] != MAGIC:
        raise ValueError("Not a QSF container")

    content, tail = blob[:-32], blob[-32:]
    if hashlib.sha256(content).digest() != tail:
        raise ValueError("Integrity check failed — container corrupted")

    flags = blob[4]
    meta_len, = struct.unpack(">I", blob[5:9])
    meta_bytes = blob[9:9 + meta_len]
    metadata = json.loads(meta_bytes)
    off = 9 + meta_len

    compressed = bool(flags & _FLAG_COMPRESSED)
    encrypted = bool(flags & _FLAG_ENCRYPTED)
    signed = bool(flags & _FLAG_SIGNED)

    kyber_ct = None
    if encrypted:
        ct_blob, off = _get_section(content, off)
        kyber_ct = _deserialize_kyber_ct(ct_blob)

    sig = None
    if signed:
        sig_blob, off = _get_section(content, off)
        sig = q_sign.Signature.from_bytes(sig_blob)

    body, off = _get_section(content, off)

    signature_valid: Optional[bool] = None
    if signer_root is not None:
        if not signed:
            raise ValueError("Container is unsigned but a signer root was required")
        signature_valid = q_sign.verify(meta_bytes + body, sig, signer_root)
        if not signature_valid:
            raise ValueError("Signature verification FAILED — do not trust this container")
    elif signed:
        signature_valid = None   # present but unchecked

    if encrypted:
        if sk is None:
            raise ValueError("Container is encrypted — Kyber private key required")
        shared = q_kyber.decapsulate(sk, kyber_ct)
        body = _keystream_xor(shared, body)

    if compressed:
        body = q_pack.decompress(body)

    return QSFContent(
        payload=body,
        metadata=metadata,
        compressed=compressed,
        encrypted=encrypted,
        signed=signed,
        signature_valid=signature_valid,
    )
