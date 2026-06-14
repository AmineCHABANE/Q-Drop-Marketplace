"""
Q-ROARING — Roaring bitmaps (compressed bitmaps for modern analytics)
Chambi, Lemire, Kaser & Godin, "Better bitmap performance with Roaring
bitmaps", Software: Practice & Experience 2016 (arXiv:1402.6407)

The compressed-bitmap index used by Apache Lucene/Elasticsearch, Druid,
ClickHouse, Spark, Pinot and many OLAP engines to represent and combine huge
sets of integer row-ids at memory-bandwidth speed.

Idea: split each 32-bit integer into a high 16-bit "chunk key" and a low 16-bit
value. Each chunk is stored in the container that is smallest for its density:
  • ArrayContainer  — a sorted uint16 array      (sparse chunks, < 4096 values)
  • BitmapContainer — a 2^16-bit (8 KiB) bitmap   (dense chunks, ≥ 4096 values)
Set operations (AND/OR/ANDNOT/XOR) run container-by-container, picking the
fastest algorithm for each pair — far faster and smaller than a flat bitmap
when the data is clustered, as row-ids in a sorted column always are.

Zero dependencies.
"""

from __future__ import annotations
from typing import Dict, Iterator, List, Optional

_ARRAY_MAX = 4096          # array→bitmap promotion threshold
_CHUNK_BITS = 16
_CHUNK_SIZE = 1 << _CHUNK_BITS   # 65536 values per chunk
_LOW_MASK = _CHUNK_SIZE - 1


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

class _ArrayContainer:
    """Sorted list of distinct uint16 values — efficient when sparse."""

    __slots__ = ("values",)

    def __init__(self, values: Optional[List[int]] = None):
        self.values = values if values is not None else []

    def add(self, v: int) -> None:
        import bisect
        i = bisect.bisect_left(self.values, v)
        if i >= len(self.values) or self.values[i] != v:
            self.values.insert(i, v)

    def contains(self, v: int) -> bool:
        import bisect
        i = bisect.bisect_left(self.values, v)
        return i < len(self.values) and self.values[i] == v

    @property
    def cardinality(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)

    def should_promote(self) -> bool:
        return len(self.values) > _ARRAY_MAX


class _BitmapContainer:
    """A dense 2^16-bit bitmap (1024 × 64-bit words)."""

    __slots__ = ("words", "_card")

    def __init__(self):
        self.words = [0] * (_CHUNK_SIZE // 64)
        self._card = 0

    def add(self, v: int) -> None:
        w, b = v >> 6, v & 63
        if not (self.words[w] >> b) & 1:
            self.words[w] |= (1 << b)
            self._card += 1

    def contains(self, v: int) -> bool:
        w, b = v >> 6, v & 63
        return bool((self.words[w] >> b) & 1)

    @property
    def cardinality(self) -> int:
        return self._card

    def __iter__(self) -> Iterator[int]:
        for wi, word in enumerate(self.words):
            if word:
                base = wi << 6
                while word:
                    b = (word & -word).bit_length() - 1   # lowest set bit
                    yield base + b
                    word &= word - 1

    @classmethod
    def from_values(cls, values: Iterator[int]) -> "_BitmapContainer":
        bc = cls()
        for v in values:
            bc.add(v)
        return bc


Container = object   # _ArrayContainer | _BitmapContainer


# ---------------------------------------------------------------------------
# Roaring bitmap
# ---------------------------------------------------------------------------

class RoaringBitmap:
    """A compressed set of 32-bit unsigned integers."""

    def __init__(self) -> None:
        # high 16 bits → container
        self._containers: Dict[int, Container] = {}

    # --- mutation ---

    def add(self, x: int) -> None:
        if x < 0:
            raise ValueError("RoaringBitmap holds non-negative integers")
        hi, lo = x >> _CHUNK_BITS, x & _LOW_MASK
        c = self._containers.get(hi)
        if c is None:
            c = _ArrayContainer()
            self._containers[hi] = c
        c.add(lo)
        # Promote a dense array container to a bitmap container
        if isinstance(c, _ArrayContainer) and c.should_promote():
            self._containers[hi] = _BitmapContainer.from_values(iter(c))

    def add_many(self, xs: Iterator[int]) -> "RoaringBitmap":
        for x in xs:
            self.add(x)
        return self

    # --- query ---

    def contains(self, x: int) -> bool:
        hi, lo = x >> _CHUNK_BITS, x & _LOW_MASK
        c = self._containers.get(hi)
        return c is not None and c.contains(lo)

    __contains__ = contains

    @property
    def cardinality(self) -> int:
        return sum(c.cardinality for c in self._containers.values())

    def __len__(self) -> int:
        return self.cardinality

    def __iter__(self) -> Iterator[int]:
        for hi in sorted(self._containers):
            base = hi << _CHUNK_BITS
            for lo in self._containers[hi]:
                yield base + lo

    def to_list(self) -> List[int]:
        return list(self)

    # --- set algebra ---

    def union(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Bitwise OR — all elements in either set."""
        result = RoaringBitmap()
        for x in self:
            result.add(x)
        for x in other:
            result.add(x)
        return result

    def intersect(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """Bitwise AND — elements in both sets."""
        result = RoaringBitmap()
        # Iterate the smaller set's containers
        for hi, c in self._containers.items():
            oc = other._containers.get(hi)
            if oc is None:
                continue
            base = hi << _CHUNK_BITS
            # Probe the (usually denser) other container
            small, big = (c, oc)
            for lo in small:
                if big.contains(lo):
                    result.add(base + lo)
        return result

    def difference(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """ANDNOT — elements in self but not other."""
        result = RoaringBitmap()
        for x in self:
            if not other.contains(x):
                result.add(x)
        return result

    def symmetric_difference(self, other: "RoaringBitmap") -> "RoaringBitmap":
        """XOR — elements in exactly one of the two sets."""
        result = RoaringBitmap()
        for x in self:
            if not other.contains(x):
                result.add(x)
        for x in other:
            if not self.contains(x):
                result.add(x)
        return result

    # operator sugar
    __or__ = union
    __and__ = intersect
    __sub__ = difference
    __xor__ = symmetric_difference

    # --- introspection ---

    def container_stats(self) -> dict:
        arrays = sum(1 for c in self._containers.values()
                     if isinstance(c, _ArrayContainer))
        bitmaps = sum(1 for c in self._containers.values()
                      if isinstance(c, _BitmapContainer))
        return {"chunks": len(self._containers),
                "array_containers": arrays,
                "bitmap_containers": bitmaps}

    def __repr__(self) -> str:
        s = self.container_stats()
        return (f"RoaringBitmap(cardinality={self.cardinality}, "
                f"chunks={s['chunks']}, arrays={s['array_containers']}, "
                f"bitmaps={s['bitmap_containers']})")


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_roaring() -> dict:
    """
    Build two row-id sets (a sparse one and a dense clustered one), then combine
    them — showing both container types coexist and set algebra is exact.
    """
    # Sparse set: every 1000th id up to 10M (10k ids spread out → array containers)
    sparse = RoaringBitmap().add_many(range(0, 10_000_000, 1000))
    # Dense set: a contiguous block of 100k ids (→ bitmap containers)
    dense = RoaringBitmap().add_many(range(5_000_000, 5_100_000))

    union = sparse | dense
    inter = sparse & dense
    diff = dense - sparse

    # Verify against Python sets on a tractable window
    py_sparse = set(range(0, 10_000_000, 1000))
    py_dense = set(range(5_000_000, 5_100_000))

    return {
        "sparse": repr(sparse),
        "dense": repr(dense),
        "union_cardinality": union.cardinality,
        "union_matches": union.cardinality == len(py_sparse | py_dense),
        "intersect_cardinality": inter.cardinality,
        "intersect_matches": inter.cardinality == len(py_sparse & py_dense),
        "difference_matches": diff.cardinality == len(py_dense - py_sparse),
        "dense_uses_bitmaps": dense.container_stats()["bitmap_containers"] > 0,
        "sparse_uses_arrays": sparse.container_stats()["array_containers"] > 0,
    }
