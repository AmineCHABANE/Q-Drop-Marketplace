"""
Q-FENWICK — Fenwick tree & segment tree (fast range queries)
  • FenwickTree (Binary Indexed Tree)  — Fenwick 1994
  • SegmentTree (generic associative)  — Bentley 1977
  • LazySegmentTree (range update)     — lazy propagation

The data structures that turn "sum/min/max over a range, with live updates"
from O(n) per query into O(log n). They sit under:
  - database aggregate indexes and OLAP cubes
  - time-series rollups and analytics dashboards
  - competitive programming (the #1 range-query tool)
  - game engines (interval bookkeeping), genomics (coverage tracks)

A Fenwick tree is the leanest structure for prefix sums + point updates; a
segment tree generalises to any associative operation and supports lazy
range updates. Both are validated here against brute force.

Zero dependencies.
"""

from __future__ import annotations
from typing import Callable, List, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Fenwick tree (Binary Indexed Tree)
# ---------------------------------------------------------------------------

class FenwickTree:
    """
    Prefix-sum structure with O(log n) point update and prefix/range queries.

    Uses the lowest-set-bit trick (`i & -i`) to hop between responsibility
    ranges — the entire elegance of the structure is in those two lines.
    """

    def __init__(self, size: int):
        self.n = size
        self._tree = [0] * (size + 1)   # 1-indexed

    @classmethod
    def from_list(cls, values: List[float]) -> "FenwickTree":
        ft = cls(len(values))
        for i, v in enumerate(values):
            ft.update(i, v)
        return ft

    def update(self, i: int, delta: float) -> None:
        """Add `delta` at index i (0-based)."""
        i += 1
        while i <= self.n:
            self._tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i: int) -> float:
        """Sum of indices [0, i] inclusive (0-based)."""
        i += 1
        s = 0.0
        while i > 0:
            s += self._tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, lo: int, hi: int) -> float:
        """Sum over [lo, hi] inclusive."""
        if lo > hi:
            return 0.0
        return self.prefix_sum(hi) - (self.prefix_sum(lo - 1) if lo > 0 else 0.0)

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# Generic segment tree (any associative operation)
# ---------------------------------------------------------------------------

class SegmentTree:
    """
    Range query for any associative combine() with an identity element.
    Point updates and range queries in O(log n).

    Example: sum (combine=+, identity=0), min (combine=min, identity=+inf),
    max, gcd, bitwise-or — anything associative.
    """

    def __init__(self, values: List[T],
                 combine: Callable[[T, T], T],
                 identity: T):
        self.n = len(values)
        self.combine = combine
        self.identity = identity
        self._t: List[T] = [identity] * (2 * self.n)
        # build: leaves at [n, 2n)
        for i, v in enumerate(values):
            self._t[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self._t[i] = combine(self._t[2 * i], self._t[2 * i + 1])

    def update(self, i: int, value: T) -> None:
        """Set index i (0-based) to `value`."""
        i += self.n
        self._t[i] = value
        i //= 2
        while i >= 1:
            self._t[i] = self.combine(self._t[2 * i], self._t[2 * i + 1])
            i //= 2

    def query(self, lo: int, hi: int) -> T:
        """Combine over [lo, hi] inclusive (0-based)."""
        res = self.identity
        l = lo + self.n
        r = hi + self.n + 1            # half-open on the right
        while l < r:
            if l & 1:
                res = self.combine(res, self._t[l])
                l += 1
            if r & 1:
                r -= 1
                res = self.combine(res, self._t[r])
            l //= 2
            r //= 2
        return res

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# Lazy segment tree (range update + range sum)
# ---------------------------------------------------------------------------

class LazySegmentTree:
    """
    Segment tree with lazy propagation supporting:
      - range_add(lo, hi, delta)  — add delta to every element in [lo, hi]
      - range_sum(lo, hi)         — sum over [lo, hi]
    both in O(log n). The lazy tags defer child updates until a query forces
    them down — the technique that makes range updates affordable.
    """

    def __init__(self, values: List[float]):
        self.n = len(values)
        self._sum = [0.0] * (4 * self.n)
        self._lazy = [0.0] * (4 * self.n)
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, node: int, lo: int, hi: int, values: List[float]) -> None:
        if lo == hi:
            self._sum[node] = values[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self._sum[node] = self._sum[2 * node] + self._sum[2 * node + 1]

    def _push_down(self, node: int, lo: int, hi: int) -> None:
        if self._lazy[node]:
            mid = (lo + hi) // 2
            for child, clo, chi in ((2 * node, lo, mid),
                                    (2 * node + 1, mid + 1, hi)):
                self._lazy[child] += self._lazy[node]
                self._sum[child] += self._lazy[node] * (chi - clo + 1)
            self._lazy[node] = 0.0

    def range_add(self, lo: int, hi: int, delta: float,
                  node: int = 1, nlo: int = 0, nhi: int = None) -> None:
        if nhi is None:
            nhi = self.n - 1
        if hi < nlo or nhi < lo:
            return
        if lo <= nlo and nhi <= hi:
            self._sum[node] += delta * (nhi - nlo + 1)
            self._lazy[node] += delta
            return
        self._push_down(node, nlo, nhi)
        mid = (nlo + nhi) // 2
        self.range_add(lo, hi, delta, 2 * node, nlo, mid)
        self.range_add(lo, hi, delta, 2 * node + 1, mid + 1, nhi)
        self._sum[node] = self._sum[2 * node] + self._sum[2 * node + 1]

    def range_sum(self, lo: int, hi: int,
                  node: int = 1, nlo: int = 0, nhi: int = None) -> float:
        if nhi is None:
            nhi = self.n - 1
        if hi < nlo or nhi < lo:
            return 0.0
        if lo <= nlo and nhi <= hi:
            return self._sum[node]
        self._push_down(node, nlo, nhi)
        mid = (nlo + nhi) // 2
        return (self.range_sum(lo, hi, 2 * node, nlo, mid)
                + self.range_sum(lo, hi, 2 * node + 1, mid + 1, nhi))

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_fenwick() -> dict:
    """
    Run all three structures against a small array and a few live updates,
    cross-checking every answer against the brute-force ground truth.
    """
    import random
    rng = random.Random(7)
    data = [rng.randint(0, 100) for _ in range(64)]

    # Fenwick prefix/range sums
    ft = FenwickTree.from_list(data)
    ft.update(10, 50)                       # data[10] += 50
    data[10] += 50
    fenwick_ok = all(
        ft.range_sum(lo, hi) == sum(data[lo:hi + 1])
        for lo, hi in [(0, 63), (5, 20), (30, 30), (0, 0)]
    )

    # Segment tree for range-min
    smin = SegmentTree(data, combine=min, identity=float("inf"))
    smin.update(15, -5)
    data[15] = -5
    min_ok = all(
        smin.query(lo, hi) == min(data[lo:hi + 1])
        for lo, hi in [(0, 63), (10, 20), (15, 15)]
    )

    # Lazy segment tree range-add + range-sum
    base = list(data)
    lazy = LazySegmentTree(base)
    lazy.range_add(8, 40, 7)                # add 7 to [8,40]
    for i in range(8, 41):
        base[i] += 7
    lazy_ok = all(
        abs(lazy.range_sum(lo, hi) - sum(base[lo:hi + 1])) < 1e-9
        for lo, hi in [(0, 63), (8, 40), (20, 50)]
    )

    return {
        "fenwick_correct": fenwick_ok,
        "segment_min_correct": min_ok,
        "lazy_range_update_correct": lazy_ok,
        "size": len(data),
    }
