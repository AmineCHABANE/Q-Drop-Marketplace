"""
q_pack.py — LZSS + canonical Huffman compression (the DEFLATE recipe)

References:
  - Ziv & Lempel, "A Universal Algorithm for Sequential Data Compression",
    IEEE Trans. Information Theory, 1977 (LZ77)
  - Storer & Szymanski, "Data Compression via Textual Substitution",
    JACM 1982 (LZSS)
  - Huffman, "A Method for the Construction of Minimum-Redundancy Codes",
    Proc. IRE, 1952
  - RFC 1951 (DEFLATE) — the same two-stage pipeline used by gzip/zlib/PNG

The pipeline (identical in spirit to gzip):
  1. LZSS — replace repeated byte sequences with (offset, length) back-references
  2. Huffman — entropy-code the resulting stream with optimal prefix codes

Why this matters for the quantum era:
  Post-quantum ciphertexts and signatures are BIG (a Kyber ciphertext is ~1KB,
  a hash-based signature several KB). Compression of the surrounding container
  matters more, not less, as cryptography gets fatter. This module is the
  compression stage of the QSF container format (src/q_qsf.py).

Format produced by compress():
  magic "QPK1" | uint32 original_len | 256 × uint8 code lengths | bitstream

Pure Python 3, zero dependencies.
"""

import heapq
import struct
from typing import Dict, List, Optional, Tuple


MAGIC = b"QPK1"
MAGIC_STORED = b"QPK0"   # raw fallback when compression would expand (RFC 1951 "stored" blocks)

# LZSS parameters
_MIN_MATCH = 3          # shorter matches cost more than literals
_MAX_MATCH = 18         # 4 bits store length - MIN_MATCH (0..15)
_WINDOW = 4096          # 12-bit offsets (1..4096)
_HASH_CHAIN_LIMIT = 32  # bound match search for O(n) worst case


# ---------------------------------------------------------------------------
# Stage 1 — LZSS
# ---------------------------------------------------------------------------

def _lzss_tokens(data: bytes) -> List:
    """
    Greedy LZSS tokenizer: literals (int 0..255) and matches (offset, length).
    A hash-chain index over 3-byte prefixes bounds the search to
    _HASH_CHAIN_LIMIT candidates per position, keeping worst case O(n).
    """
    tokens = []
    n = len(data)
    chains: Dict[int, List[int]] = {}

    def index_pos(p: int) -> None:
        if p + _MIN_MATCH <= n:
            key = data[p] << 16 | data[p + 1] << 8 | data[p + 2]
            bucket = chains.setdefault(key, [])
            bucket.insert(0, p)
            if len(bucket) > _HASH_CHAIN_LIMIT:
                bucket.pop()

    i = 0
    while i < n:
        best_len, best_off = 0, 0
        if i + _MIN_MATCH <= n:
            key = data[i] << 16 | data[i + 1] << 8 | data[i + 2]
            for pos in chains.get(key, ()):
                if i - pos > _WINDOW:
                    break
                length = 0
                max_len = min(_MAX_MATCH, n - i)
                while length < max_len and data[pos + length] == data[i + length]:
                    length += 1
                if length > best_len:
                    best_len, best_off = length, i - pos
                    if length == _MAX_MATCH:
                        break

        if best_len >= _MIN_MATCH:
            tokens.append((best_off, best_len))
            for p in range(i, i + best_len):
                index_pos(p)
            i += best_len
        else:
            tokens.append(data[i])
            index_pos(i)
            i += 1
    return tokens


def _serialize_tokens(tokens: List) -> bytes:
    """
    LZSS wire form: groups of 8 tokens preceded by a flag byte.
    Flag bit 0 = literal (1 byte), bit 1 = match (2 bytes:
    12-bit offset-1, 4-bit length-MIN_MATCH).
    """
    out = bytearray()
    for g in range(0, len(tokens), 8):
        group = tokens[g:g + 8]
        flags = 0
        body = bytearray()
        for bit, tok in enumerate(group):
            if isinstance(tok, tuple):
                flags |= 1 << bit
                off, length = tok
                packed = ((off - 1) << 4) | (length - _MIN_MATCH)
                body += struct.pack(">H", packed)
            else:
                body.append(tok)
        out.append(flags)
        out += body
    return bytes(out)


def _deserialize_tokens(blob: bytes, orig_len: int) -> bytes:
    """Decode the LZSS wire form back into the original bytes."""
    out = bytearray()
    i = 0
    n = len(blob)
    while i < n and len(out) < orig_len:
        flags = blob[i]
        i += 1
        for bit in range(8):
            if len(out) >= orig_len or i >= n:
                break
            if flags >> bit & 1:
                packed, = struct.unpack(">H", blob[i:i + 2])
                i += 2
                off = (packed >> 4) + 1
                length = (packed & 0xF) + _MIN_MATCH
                start = len(out) - off
                for k in range(length):     # may self-overlap (run encoding)
                    out.append(out[start + k])
            else:
                out.append(blob[i])
                i += 1
    return bytes(out)


# ---------------------------------------------------------------------------
# Stage 2 — canonical Huffman
# ---------------------------------------------------------------------------

def _code_lengths(data: bytes) -> List[int]:
    """Build Huffman code lengths (max 255) for all 256 byte symbols."""
    freq = [0] * 256
    for b in data:
        freq[b] += 1

    # heap of (weight, tiebreak, symbols-with-depths)
    heap = []
    tie = 0
    for sym, f in enumerate(freq):
        if f:
            heap.append((f, tie, {sym: 0}))
            tie += 1
    if not heap:
        return [0] * 256
    if len(heap) == 1:
        # single distinct symbol: give it a 1-bit code
        lengths = [0] * 256
        lengths[next(iter(heap[0][2]))] = 1
        return lengths

    heapq.heapify(heap)
    while len(heap) > 1:
        w1, _, d1 = heapq.heappop(heap)
        w2, _, d2 = heapq.heappop(heap)
        merged = {s: d + 1 for s, d in d1.items()}
        merged.update({s: d + 1 for s, d in d2.items()})
        heapq.heappush(heap, (w1 + w2, tie, merged))
        tie += 1

    lengths = [0] * 256
    for sym, depth in heap[0][2].items():
        lengths[sym] = depth
    return lengths


def _canonical_codes(lengths: List[int]) -> Dict[int, Tuple[int, int]]:
    """
    Assign canonical Huffman codes from code lengths (RFC 1951 §3.2.2).
    Returns {symbol: (code, length)}. Canonical codes let the decoder
    rebuild the exact table from lengths alone — no tree in the header.
    """
    pairs = sorted((l, s) for s, l in enumerate(lengths) if l > 0)
    codes: Dict[int, Tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for length, sym in pairs:
        code <<= (length - prev_len)
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


class _BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code: int, length: int) -> None:
        self.acc = (self.acc << length) | code
        self.nbits += length
        while self.nbits >= 8:
            self.nbits -= 8
            self.buf.append((self.acc >> self.nbits) & 0xFF)

    def finish(self) -> bytes:
        if self.nbits:
            self.buf.append((self.acc << (8 - self.nbits)) & 0xFF)
            self.nbits = 0
        return bytes(self.buf)


class _BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read_bit(self) -> int:
        if self.nbits == 0:
            self.acc = self.data[self.pos]
            self.pos += 1
            self.nbits = 8
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1


def _huffman_encode(data: bytes) -> Tuple[bytes, bytes]:
    """Returns (code_length_table[256 bytes], bitstream)."""
    lengths = _code_lengths(data)
    codes = _canonical_codes(lengths)
    w = _BitWriter()
    for b in data:
        code, length = codes[b]
        w.write(code, length)
    return bytes(lengths), w.finish()


def _huffman_decode(lengths: bytes, bitstream: bytes, n_symbols: int) -> bytes:
    """Decode n_symbols bytes using canonical codes rebuilt from lengths."""
    codes = _canonical_codes(list(lengths))
    # decode table: (length, code) → symbol
    table = {(l, c): s for s, (c, l) in codes.items()}
    r = _BitReader(bitstream)
    out = bytearray()
    code = 0
    length = 0
    while len(out) < n_symbols:
        code = (code << 1) | r.read_bit()
        length += 1
        sym = table.get((length, code))
        if sym is not None:
            out.append(sym)
            code = 0
            length = 0
    return bytes(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress(data: bytes) -> bytes:
    """
    Compress bytes with LZSS + canonical Huffman.

    Layout: MAGIC | uint32 orig_len | uint32 lzss_len | 256B lengths | bits
    Empty input compresses to just the header.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("compress() expects bytes")
    data = bytes(data)
    if not data:
        return MAGIC + struct.pack(">II", 0, 0)

    tokens = _lzss_tokens(data)
    lzss = _serialize_tokens(tokens)
    lengths, bits = _huffman_encode(lzss)
    packed = MAGIC + struct.pack(">II", len(data), len(lzss)) + lengths + bits
    if len(packed) >= len(data) + 4:
        return MAGIC_STORED + data
    return packed


def decompress(blob: bytes) -> bytes:
    """Inverse of compress(). Raises ValueError on bad magic / truncation."""
    if blob[:4] == MAGIC_STORED:
        return blob[4:]
    if blob[:4] != MAGIC:
        raise ValueError("Not a QPK1 stream")
    orig_len, lzss_len = struct.unpack(">II", blob[4:12])
    if orig_len == 0:
        return b""
    lengths = blob[12:268]
    bitstream = blob[268:]
    lzss = _huffman_decode(lengths, bitstream, lzss_len)
    out = _deserialize_tokens(lzss, orig_len)
    if len(out) != orig_len:
        raise ValueError("Truncated or corrupt QPK1 stream")
    return out


def ratio(data: bytes) -> float:
    """Convenience: compressed size / original size (lower is better)."""
    if not data:
        return 1.0
    return len(compress(data)) / len(data)
