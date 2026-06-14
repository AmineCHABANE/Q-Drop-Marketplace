"""
Q-TOPK — Count-Min Sketch & heavy hitters (frequency estimation on streams)
Cormode & Muthukrishnan, "An Improved Data Stream Summary: The Count-Min
Sketch and its Applications", LATIN/J. Algorithms 2005

Estimates how often each item appears in a stream using fixed, tiny memory —
independent of how many distinct items flow past. Overestimates only, never
underestimates, with a tunable error bound. The structure behind:
  - "top trending" / heavy-hitter detection (search, ads, abuse)
  - network telemetry (per-flow byte counts at line rate)
  - database query optimizers (frequency estimates for cardinality)
  - Redis-Bloom CMS, Apache Spark, and streaming analytics generally

Width w = ⌈e/ε⌉ and depth d = ⌈ln(1/δ)⌉ give, for total count N, an estimate
within ε·N of the true count with probability ≥ 1−δ. Paired here with a
heavy-hitter tracker that surfaces the most frequent items in one pass.

Zero dependencies.
"""

from __future__ import annotations
import hashlib
import heapq
import math
from typing import Any, Dict, Hashable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------

class CountMinSketch:
    """
    A d×w table of counters. Each item hashes to one column per row; add()
    increments those d cells, estimate() returns the MINIMUM of them (the cell
    least affected by collisions). Hence estimates are upper bounds on truth.
    """

    def __init__(self, epsilon: float = 0.001, delta: float = 0.01):
        """
        :param epsilon: relative error (fraction of total count) — smaller w grows
        :param delta:   failure probability — smaller → more rows
        """
        self.w = max(2, math.ceil(math.e / epsilon))
        self.d = max(1, math.ceil(math.log(1.0 / delta)))
        self.epsilon = epsilon
        self.delta = delta
        self._table: List[List[int]] = [[0] * self.w for _ in range(self.d)]
        self._total = 0

    def _hashes(self, item: Hashable) -> List[int]:
        raw = item if isinstance(item, bytes) else str(item).encode()
        out = []
        for row in range(self.d):
            h = hashlib.blake2b(raw, digest_size=8,
                                salt=row.to_bytes(2, "big")).digest()
            out.append(int.from_bytes(h, "big") % self.w)
        return out

    def add(self, item: Hashable, count: int = 1) -> None:
        self._total += count
        for row, col in enumerate(self._hashes(item)):
            self._table[row][col] += count

    def estimate(self, item: Hashable) -> int:
        """Estimated frequency of `item` (an upper bound on the true count)."""
        return min(self._table[row][col]
                   for row, col in enumerate(self._hashes(item)))

    @property
    def total(self) -> int:
        return self._total

    @property
    def error_bound(self) -> float:
        """Additive error guarantee: estimate ≤ true + error_bound (w.h.p.)."""
        return self.epsilon * self._total

    @property
    def memory_cells(self) -> int:
        return self.d * self.w

    def __repr__(self) -> str:
        return (f"CountMinSketch(d={self.d}, w={self.w}, "
                f"cells={self.memory_cells}, total={self._total})")


# ---------------------------------------------------------------------------
# Heavy hitters (top-k frequent items in one pass)
# ---------------------------------------------------------------------------

class HeavyHitters:
    """
    Streaming top-k: a Count-Min Sketch estimates frequencies while a bounded
    min-heap keeps the k items with the highest estimates seen so far. One pass,
    fixed memory, no need to store the stream.
    """

    def __init__(self, k: int = 10, epsilon: float = 0.001, delta: float = 0.01):
        self.k = k
        self._cms = CountMinSketch(epsilon, delta)
        self._heap: List[Tuple[int, Any]] = []     # (estimate, item) min-heap
        self._in_heap: Dict[Any, int] = {}         # item -> current estimate

    def add(self, item: Hashable, count: int = 1) -> None:
        self._cms.add(item, count)
        est = self._cms.estimate(item)

        if item in self._in_heap:
            # Refresh its estimate by rebuilding the heap lazily
            self._in_heap[item] = est
            self._rebuild()
            return

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (est, item))
            self._in_heap[item] = est
        elif est > self._heap[0][0]:
            _, evicted = heapq.heapreplace(self._heap, (est, item))
            del self._in_heap[evicted]
            self._in_heap[item] = est

    def _rebuild(self) -> None:
        self._heap = [(e, it) for it, e in self._in_heap.items()]
        heapq.heapify(self._heap)

    def top(self) -> List[Tuple[Any, int]]:
        """Return up to k (item, estimated_count) pairs, most frequent first."""
        return sorted(self._in_heap.items(), key=lambda kv: -kv[1])

    def estimate(self, item: Hashable) -> int:
        return self._cms.estimate(item)

    def __repr__(self) -> str:
        return f"HeavyHitters(k={self.k}, tracked={len(self._in_heap)})"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_topk() -> dict:
    """
    Stream a Zipf-like distribution of events through a Count-Min Sketch + heavy
    hitter tracker in fixed memory, then verify: estimates never underestimate,
    stay within the error bound, and the recovered top-k matches the true top-k.
    """
    import random
    rng = random.Random(42)

    # Build a skewed stream: a few very frequent items + a long tail
    stream: List[str] = []
    hot = {f"hot{i}": (5000 // (i + 1)) for i in range(10)}   # heavy hitters
    for item, freq in hot.items():
        stream += [item] * freq
    for _ in range(20000):                                    # noisy long tail
        stream.append(f"tail{rng.randint(0, 5000)}")
    rng.shuffle(stream)

    # Exact ground truth (only to validate — the whole point is to avoid storing this)
    from collections import Counter
    true = Counter(stream)

    cms = CountMinSketch(epsilon=0.0005, delta=0.01)
    hh = HeavyHitters(k=10, epsilon=0.0005, delta=0.01)
    for item in stream:
        cms.add(item)
        hh.add(item)

    # Estimates must be >= truth (one-sided error) and within the additive bound
    never_under = all(cms.estimate(it) >= true[it] for it in hot)
    within_bound = all(cms.estimate(it) - true[it] <= cms.error_bound for it in hot)

    recovered_top = [it for it, _ in hh.top()]
    true_top = [it for it, _ in true.most_common(10)]
    topk_recall = len(set(recovered_top) & set(true_top)) / 10

    return {
        "sketch": repr(cms),
        "stream_length": len(stream),
        "distinct_items": len(true),
        "estimates_never_underestimate": never_under,
        "within_error_bound": within_bound,
        "error_bound": round(cms.error_bound, 1),
        "topk_recall": topk_recall,
        "recovered_top5": recovered_top[:5],
    }
