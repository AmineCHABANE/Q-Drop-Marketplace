"""
Q-BLOOM — Probabilistic membership filters
  1. Classic Bloom filter        — space-optimal, false-positive only
  2. Counting Bloom filter       — supports deletion via saturating counters
  3. Xor filter (Binary Fuse 8)  — ~9% more compact than Bloom, static
  4. Cuckoo filter               — deletable, cache-friendly, better FPR
  5. HyperLogLog                 — approximate cardinality (count-distinct)
  6. MinHash / LSH                — similarity estimation + locality-sensitive hashing

Applications:
  - Bloom: LSM-tree key lookups before SSTable I/O (RocksDB, Cassandra)
  - Counting: network routing tables, cache eviction policies
  - Xor/Cuckoo: CDN edge caches, Chrome Safe Browsing
  - HyperLogLog: Redis PFADD/PFCOUNT, BigQuery COUNT DISTINCT
  - MinHash: document deduplication, plagiarism detection, collaborative filtering

References:
  Bloom 1970; Fan et al. 2000 (Counting); Graaf & Bhatt 2022 (Xor/Binary Fuse);
  Pagh & Rodler 2004 (Cuckoo); Flajolet & Martin 1985 (HyperLogLog);
  Broder 1997 (MinHash)
"""

from __future__ import annotations
import hashlib
import math
import random
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _hash_bytes(item: Any) -> bytes:
    raw = item if isinstance(item, (bytes, bytearray)) else str(item).encode()
    return hashlib.sha256(raw).digest()

def _multi_hash(item: Any, k: int, m: int) -> List[int]:
    """Return k independent hash values in [0, m) using SHA-256 + counter."""
    result = []
    h = _hash_bytes(item)
    for i in range(k):
        # Mix in counter i via XOR of the first 8 bytes with i
        mixed = (int.from_bytes(h[:8], "little") ^ (i * 0x9e3779b97f4a7c15)) & ((1 << 64) - 1)
        result.append(mixed % m)
    return result

def _hash64(item: Any, seed: int = 0) -> int:
    raw = item if isinstance(item, (bytes, bytearray)) else str(item).encode()
    mixed = struct.pack("<Q", seed) + raw
    return int.from_bytes(hashlib.sha256(mixed).digest()[:8], "little")

def _hash32(item: Any, seed: int = 0) -> int:
    return _hash64(item, seed) & 0xFFFF_FFFF


# ---------------------------------------------------------------------------
# 1. Classic Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Space-optimal probabilistic set membership.

    False positives possible (rate ≤ fpr), false negatives impossible.
    No deletion support.

    Optimal parameters:
      m = -n·ln(fpr) / (ln2)²  bits
      k = (m/n)·ln2             hash functions
    """

    def __init__(self, capacity: int, fpr: float = 0.01):
        """
        :param capacity: expected number of distinct items
        :param fpr: desired false-positive rate (0 < fpr < 1)
        """
        assert 0 < fpr < 1
        self.capacity = capacity
        self.fpr      = fpr
        m = max(1, int(-capacity * math.log(fpr) / (math.log(2) ** 2)))
        self.m = m
        self.k = max(1, int(m / capacity * math.log(2)))
        self._bits = bytearray((m + 7) // 8)
        self._count = 0

    def add(self, item: Any) -> None:
        for h in _multi_hash(item, self.k, self.m):
            self._bits[h >> 3] |= 1 << (h & 7)
        self._count += 1

    def __contains__(self, item: Any) -> bool:
        return all(
            self._bits[h >> 3] & (1 << (h & 7))
            for h in _multi_hash(item, self.k, self.m)
        )

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set (approaches optimal FPR as fill → k/m·ln2)."""
        ones = sum(bin(b).count("1") for b in self._bits)
        return ones / self.m

    @property
    def estimated_fpr(self) -> float:
        """Empirical false-positive rate given current fill."""
        return (self.fill_ratio) ** self.k

    @property
    def size_bytes(self) -> int:
        return len(self._bits)

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (f"BloomFilter(n≈{self._count}, m={self.m} bits, "
                f"k={self.k}, fpr≈{self.estimated_fpr:.4f})")

    def merge(self, other: "BloomFilter") -> "BloomFilter":
        """Union: OR the bit arrays (same parameters required)."""
        if self.m != other.m or self.k != other.k:
            raise ValueError("Cannot merge Bloom filters with different parameters")
        merged = BloomFilter.__new__(BloomFilter)
        merged.capacity = self.capacity + other.capacity
        merged.fpr = self.fpr
        merged.m = self.m
        merged.k = self.k
        merged._bits = bytearray(a | b for a, b in zip(self._bits, other._bits))
        merged._count = self._count + other._count
        return merged


# ---------------------------------------------------------------------------
# 2. Counting Bloom Filter (supports deletion)
# ---------------------------------------------------------------------------

class CountingBloomFilter:
    """
    Bloom filter with 4-bit saturating counters per slot.
    Supports both add() and remove().

    Counter saturation at 15 prevents overflow under repeated adds;
    deletion of never-added items may cause false negatives (use carefully).
    """

    _MAX = 0x0F   # 4-bit counter max

    def __init__(self, capacity: int, fpr: float = 0.01):
        self.capacity = capacity
        self.fpr      = fpr
        m = max(1, int(-capacity * math.log(fpr) / (math.log(2) ** 2)))
        self.m = m
        self.k = max(1, int(m / capacity * math.log(2)))
        # Each byte holds two 4-bit counters
        self._counters = bytearray((m + 1) // 2)
        self._count = 0

    def _get(self, slot: int) -> int:
        byte = self._counters[slot >> 1]
        return (byte >> 4) if (slot & 1) else (byte & 0x0F)

    def _inc(self, slot: int) -> None:
        idx, hi = slot >> 1, slot & 1
        byte = self._counters[idx]
        nibble = (byte >> 4) if hi else (byte & 0x0F)
        if nibble < self._MAX:
            nibble += 1
        if hi:
            self._counters[idx] = (nibble << 4) | (byte & 0x0F)
        else:
            self._counters[idx] = (byte & 0xF0) | nibble

    def _dec(self, slot: int) -> None:
        idx, hi = slot >> 1, slot & 1
        byte = self._counters[idx]
        nibble = (byte >> 4) if hi else (byte & 0x0F)
        if nibble > 0:
            nibble -= 1
        if hi:
            self._counters[idx] = (nibble << 4) | (byte & 0x0F)
        else:
            self._counters[idx] = (byte & 0xF0) | nibble

    def add(self, item: Any) -> None:
        for h in _multi_hash(item, self.k, self.m):
            self._inc(h)
        self._count += 1

    def remove(self, item: Any) -> None:
        if item not in self:
            return   # don't underflow for items never added
        for h in _multi_hash(item, self.k, self.m):
            self._dec(h)
        self._count -= 1

    def __contains__(self, item: Any) -> bool:
        return all(self._get(h) > 0 for h in _multi_hash(item, self.k, self.m))

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"CountingBloomFilter(n≈{self._count}, m={self.m}, k={self.k})"


# ---------------------------------------------------------------------------
# 3. Xor Filter — Binary Fuse 8 variant
# ---------------------------------------------------------------------------

class XorFilter:
    """
    Static membership filter using a 3-way XOR fingerprint structure.

    More compact than Bloom for the same FPR (~9 bits/element at 0.39% FPR vs ~9.6).
    Build once, query many times; no deletion.

    Construction: peeling algorithm — find slots with exactly one key,
    assign fingerprint there, propagate XOR constraints to neighbours until
    all keys are placed.  Lookup: F[h0(x)] ^ F[h1(x)] ^ F[h2(x)] == fp(x).

    Algorithm: Xor8 / Binary Fuse 8 (Graf & Lemire 2020, arXiv:1912.08258)
    """

    _BITS = 8   # 8-bit fingerprints

    def __init__(self) -> None:
        self._size: int = 0
        self._seed: int = 0
        self._fp: bytearray = bytearray()
        self._seg: int = 0

    # Three segment-based hash functions
    def _h0(self, key: int, seg: int) -> int:
        return key % seg

    def _h1(self, key: int, seg: int) -> int:
        return seg + ((key >> 21) % seg)

    def _h2(self, key: int, seg: int) -> int:
        return 2 * seg + ((key >> 42) % seg)

    def _slots(self, key: int, seg: int) -> Tuple[int, int, int]:
        return self._h0(key, seg), self._h1(key, seg), self._h2(key, seg)

    def build(self, items: Iterable[Any], seed: int = 0) -> None:
        """Construct the filter from an iterable of distinct items (tries multiple seeds)."""
        raw = list(items)
        n = len(raw)
        if n == 0:
            self._size = 0
            self._fp = bytearray()
            return

        for attempt in range(32):
            actual_seed = seed + attempt
            keys = [_hash64(x, actual_seed) for x in raw]
            # Segment: slightly larger than n/3 to ensure ~95% fill → successful peel
            seg = max(2, int(math.ceil(n * 1.23 / 3)) + 1)
            ok, fp = self._peel(keys, seg)
            if ok:
                self._seed = actual_seed
                self._seg = seg
                self._size = n
                self._fp = fp
                return
        raise RuntimeError("XorFilter: peeling failed after 32 attempts — duplicate keys?")

    def _peel(self, keys: List[int], seg: int) -> Tuple[bool, bytearray]:
        """Peeling-based construction.  Returns (success, fingerprint_array)."""
        cap = 3 * seg
        # count[slot] = how many keys map here; xor_at[slot] = XOR of those key hashes
        count  = [0] * cap
        xor_at = [0] * cap

        for key in keys:
            for s in self._slots(key, seg):
                count[s]  += 1
                xor_at[s] ^= key

        # Pure slots: exactly one key — start peeling queue
        queue = [i for i in range(cap) if count[i] == 1]
        order: List[Tuple[int, int]] = []   # (slot, key) peel order

        while queue:
            slot = queue.pop()
            if count[slot] != 1:
                continue
            key = xor_at[slot]   # the unique key in this slot
            order.append((slot, key))
            for s in self._slots(key, seg):
                if s != slot:
                    count[s]  -= 1
                    xor_at[s] ^= key
                    if count[s] == 1:
                        queue.append(s)

        if len(order) != len(keys):
            return False, bytearray()

        # Assign fingerprints in reverse peel order
        fp = bytearray(cap)
        for slot, key in reversed(order):
            s0, s1, s2 = self._slots(key, seg)
            target = key & 0xFF
            # fp[slot] = target XOR fp[other_two_slots] (fp[slot] still 0 here)
            others = [s for s in (s0, s1, s2) if s != slot]
            fp[slot] = target ^ fp[others[0]] ^ fp[others[1]]

        return True, fp

    def __contains__(self, item: Any) -> bool:
        if not self._fp or self._size == 0:
            return False
        key  = _hash64(item, self._seed)
        seg  = self._seg
        s0, s1, s2 = self._slots(key, seg)
        return (self._fp[s0] ^ self._fp[s1] ^ self._fp[s2]) == (key & 0xFF)

    @classmethod
    def from_items(cls, items: Iterable[Any], seed: int = 0) -> "XorFilter":
        f = cls()
        f.build(items, seed)
        return f

    @property
    def size_bytes(self) -> int:
        return len(self._fp)

    def __repr__(self) -> str:
        if not self._fp:
            return "XorFilter(empty)"
        bpe = self.size_bytes * 8 / max(1, self._size)
        return f"XorFilter(n={self._size}, bytes={self.size_bytes}, bits/elem={bpe:.1f})"


# ---------------------------------------------------------------------------
# 4. Cuckoo Filter
# ---------------------------------------------------------------------------

class CuckooFilter:
    """
    Cache-friendly filter supporting insertion, lookup, and deletion.

    Each bucket holds `_BUCKET_SIZE` fingerprints of `_FP_BITS` bits.
    On collision: cuckoo-kick a random occupant to its alternate bucket.
    Max 500 kicks before reporting full (load factor ~95%).

    FPR ≈ 2·b/2^f  where b=bucket_size, f=fingerprint_bits.
    """

    _BUCKET_SIZE = 4
    _FP_BITS     = 8    # 8-bit fingerprints → ~3% FPR for b=4
    _FP_MASK     = (1 << _FP_BITS) - 1
    _MAX_KICKS   = 500

    def __init__(self, capacity: int):
        n_buckets = max(4, 1 << math.ceil(math.log2(capacity / self._BUCKET_SIZE + 1)))
        self._n = n_buckets
        # Stored as list-of-lists of 8-bit ints (0 = empty slot)
        self._buckets: List[List[int]] = [[] for _ in range(n_buckets)]
        self._count = 0
        self._rng = random.Random(0)

    def _fp(self, item: Any) -> int:
        h = _hash64(item)
        fp = (h >> 32) & self._FP_MASK
        return fp if fp != 0 else 1   # 0 reserved for empty

    def _i1(self, item: Any) -> int:
        return _hash64(item) % self._n

    def _i2(self, i1: int, fp: int) -> int:
        # Partial-key cuckoo hashing: i2 = i1 XOR hash(fp)
        return (i1 ^ _hash32(fp)) % self._n

    def add(self, item: Any) -> bool:
        """Insert item.  Returns False if filter is full."""
        fp = self._fp(item)
        i1 = self._i1(item)
        i2 = self._i2(i1, fp)

        for idx in (i1, i2):
            if len(self._buckets[idx]) < self._BUCKET_SIZE:
                self._buckets[idx].append(fp)
                self._count += 1
                return True

        # Kick
        idx = self._rng.choice([i1, i2])
        for _ in range(self._MAX_KICKS):
            slot = self._rng.randrange(len(self._buckets[idx]))
            fp, self._buckets[idx][slot] = self._buckets[idx][slot], fp
            idx = self._i2(idx, fp)
            if len(self._buckets[idx]) < self._BUCKET_SIZE:
                self._buckets[idx].append(fp)
                self._count += 1
                return True
        return False   # filter full

    def __contains__(self, item: Any) -> bool:
        fp = self._fp(item)
        i1 = self._i1(item)
        i2 = self._i2(i1, fp)
        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def remove(self, item: Any) -> bool:
        """Delete one occurrence of item.  Returns False if not found."""
        fp = self._fp(item)
        i1 = self._i1(item)
        i2 = self._i2(i1, fp)
        for idx in (i1, i2):
            if fp in self._buckets[idx]:
                self._buckets[idx].remove(fp)
                self._count -= 1
                return True
        return False

    @property
    def load_factor(self) -> float:
        total = self._n * self._BUCKET_SIZE
        return self._count / total if total > 0 else 0.0

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (f"CuckooFilter(n={self._count}, buckets={self._n}, "
                f"load={self.load_factor:.2f})")


# ---------------------------------------------------------------------------
# 5. HyperLogLog — cardinality estimation
# ---------------------------------------------------------------------------

class HyperLogLog:
    """
    Approximate count-distinct (cardinality estimator).

    Error ≈ 1.04 / √m  where m = 2^b registers.
    m=1024 (b=10) → ~3.25% error.
    Uses bias correction for small and large ranges.

    Redis PFADD / PFCOUNT uses b=14 (16 384 registers) → ~0.81% error.
    """

    def __init__(self, b: int = 10):
        """b: number of register bits (4 ≤ b ≤ 16).  m = 2^b registers."""
        assert 4 <= b <= 16
        self.b = b
        self.m = 1 << b
        self._regs = bytearray(self.m)   # 8-bit max run-length per register

    def add(self, item: Any) -> None:
        h = _hash64(item)
        # First b bits → register index; remaining 64-b bits → leading zeros
        idx = h >> (64 - self.b)
        w   = h << self.b | (1 << (self.b - 1))  # push b bits off top
        # Count leading zeros + 1 in the remaining 64 bits
        rho = 1
        while rho <= 64 and not (w >> (64 - rho)) & 1:
            rho += 1
        if rho > self._regs[idx]:
            self._regs[idx] = rho

    def count(self) -> int:
        """Return estimated cardinality."""
        m = self.m
        # Raw harmonic mean estimate
        alpha = {4: 0.673, 5: 0.697, 6: 0.709}.get(self.b, 0.7213 / (1 + 1.079 / m))
        Z = sum(2.0 ** (-r) for r in self._regs)
        raw = alpha * m * m / Z

        # Small-range correction (linear counting)
        V = self._regs.count(0)
        if raw <= 2.5 * m and V > 0:
            raw = m * math.log(m / V)

        # Large-range correction
        if raw > (1 << 32) / 30:
            raw = -(1 << 32) * math.log(1 - raw / (1 << 32))

        return int(raw)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Union: take the max of each register (union cardinality estimate)."""
        if self.b != other.b:
            raise ValueError("Cannot merge HyperLogLog with different precision")
        merged = HyperLogLog(self.b)
        for i in range(self.m):
            merged._regs[i] = max(self._regs[i], other._regs[i])
        return merged

    @property
    def size_bytes(self) -> int:
        return len(self._regs)

    def __repr__(self) -> str:
        return f"HyperLogLog(b={self.b}, m={self.m}, estimate={self.count()})"


# ---------------------------------------------------------------------------
# 6. MinHash / Locality-Sensitive Hashing
# ---------------------------------------------------------------------------

class MinHash:
    """
    MinHash sketch for Jaccard similarity estimation.

    Given sets A and B, P(min_hash(A) == min_hash(B)) = |A∩B| / |A∪B|.
    Uses n_hashes independent hash functions (simulated via double-hashing).

    Applications:
      - Near-duplicate document detection
      - Collaborative filtering (users ≈ item sets)
      - Genome sequence comparison
    """

    def __init__(self, n_hashes: int = 128, seed: int = 0):
        self.n    = n_hashes
        self.seed = seed
        self._mins = [0xFFFFFFFFFFFFFFFF] * n_hashes

    def update(self, item: Any) -> None:
        """Add an item (element of the set) to the sketch."""
        for i in range(self.n):
            h = _hash64(item, self.seed + i)
            if h < self._mins[i]:
                self._mins[i] = h

    def update_set(self, items: Iterable[Any]) -> None:
        for item in items:
            self.update(item)

    def similarity(self, other: "MinHash") -> float:
        """Estimate Jaccard similarity between this sketch and another."""
        if self.n != other.n:
            raise ValueError("Cannot compare MinHash sketches of different sizes")
        agree = sum(1 for a, b in zip(self._mins, other._mins) if a == b)
        return agree / self.n

    @classmethod
    def from_set(cls, items: Iterable[Any], n_hashes: int = 128,
                 seed: int = 0) -> "MinHash":
        m = cls(n_hashes, seed)
        m.update_set(items)
        return m

    def __repr__(self) -> str:
        return f"MinHash(n={self.n})"


class LSHBand:
    """
    Locality-Sensitive Hashing: band-based approximate nearest-neighbour search.

    Divides MinHash sketch into `bands` bands of `rows` elements each.
    Two items collide in a band (and are candidate neighbours) with probability:
      P(collision) = 1 - (1 - s^rows)^bands
    where s = Jaccard similarity.

    Threshold (where P≈0.5): s* ≈ (1/bands)^(1/rows).
    """

    def __init__(self, n_hashes: int = 128, bands: int = 16):
        assert n_hashes % bands == 0, "n_hashes must be divisible by bands"
        self.n_hashes = n_hashes
        self.bands    = bands
        self.rows     = n_hashes // bands
        self._buckets: List[Dict[int, List[Any]]] = [{} for _ in range(bands)]
        self._items:   Dict[Any, MinHash] = {}

    def add(self, key: Any, sketch: MinHash) -> None:
        self._items[key] = sketch
        for b in range(self.bands):
            band_hash = _hash64(
                b"".join(struct.pack("<Q", v)
                          for v in sketch._mins[b * self.rows:(b + 1) * self.rows]),
                seed=b,
            )
            band_hash &= 0xFFFF_FFFF_FFFF_FFFF
            self._buckets[b].setdefault(band_hash, []).append(key)

    def candidates(self, sketch: MinHash) -> Set[Any]:
        """Return all keys that share at least one band bucket with `sketch`."""
        result: Set[Any] = set()
        for b in range(self.bands):
            band_hash = _hash64(
                b"".join(struct.pack("<Q", v)
                          for v in sketch._mins[b * self.rows:(b + 1) * self.rows]),
                seed=b,
            )
            band_hash &= 0xFFFF_FFFF_FFFF_FFFF
            for key in self._buckets[b].get(band_hash, []):
                result.add(key)
        return result

    def query(self, sketch: MinHash, threshold: float = 0.5) -> List[Tuple[Any, float]]:
        """Return candidate keys with estimated similarity ≥ threshold."""
        cands = self.candidates(sketch)
        results = []
        for key in cands:
            sim = sketch.similarity(self._items[key])
            if sim >= threshold:
                results.append((key, sim))
        return sorted(results, key=lambda x: -x[1])

    @property
    def threshold(self) -> float:
        return (1 / self.bands) ** (1 / self.rows)

    def __repr__(self) -> str:
        return (f"LSHBand(n={self.n_hashes}, bands={self.bands}, "
                f"rows={self.rows}, threshold≈{self.threshold:.2f})")


# ---------------------------------------------------------------------------
# Benchmark / demo helpers
# ---------------------------------------------------------------------------

def benchmark_bloom(n: int = 100_000, fpr: float = 0.01) -> dict:
    bf = BloomFilter(n, fpr)
    items = [f"key:{i}" for i in range(n)]
    for item in items:
        bf.add(item)

    # False-positive test with non-member items
    fp_count = sum(1 for i in range(n, 2 * n) if f"key:{i}" in bf)
    actual_fpr = fp_count / n

    return {
        "type":           "BloomFilter",
        "n":              n,
        "target_fpr":     fpr,
        "actual_fpr":     actual_fpr,
        "size_bytes":     bf.size_bytes,
        "bits_per_item":  bf.size_bytes * 8 / n,
        "k":              bf.k,
    }

def benchmark_hll(n: int = 100_000, b: int = 10) -> dict:
    hll = HyperLogLog(b)
    for i in range(n):
        hll.add(f"item:{i}")
    estimate = hll.count()
    error = abs(estimate - n) / n
    return {
        "type":       "HyperLogLog",
        "actual_n":   n,
        "estimated":  estimate,
        "rel_error":  error,
        "size_bytes": hll.size_bytes,
    }

def benchmark_minhash(n_docs: int = 20, words_per_doc: int = 200,
                      seed: int = 42) -> dict:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(1000)]

    docs = {
        f"doc{i}": set(rng.choices(vocab, k=words_per_doc))
        for i in range(n_docs)
    }

    sketches = {k: MinHash.from_set(v) for k, v in docs.items()}

    # Compare first doc against all others
    d0_key = "doc0"
    d0_real = docs[d0_key]
    d0_mh   = sketches[d0_key]

    results = []
    for key, real_set in docs.items():
        if key == d0_key:
            continue
        real_jac = len(d0_real & real_set) / len(d0_real | real_set)
        est_jac  = d0_mh.similarity(sketches[key])
        results.append({"key": key, "real": round(real_jac, 3), "est": round(est_jac, 3)})

    return {"type": "MinHash", "docs": n_docs, "comparisons": results[:5]}
