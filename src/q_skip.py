"""
Q-SKIP — Skip list (the data structure behind Redis sorted sets / ZSET)
Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees", CACM 1990

A skip list is a tower of linked lists: the bottom level links every element in
order; each higher level is an express lane that skips roughly half the nodes.
Search, insert and delete are all O(log n) *expected*, with no rotations and far
simpler code than a red-black or AVL tree — which is exactly why Redis chose it
for ZSET (ZADD / ZRANGE / ZRANK), and why LevelDB uses it for the MemTable.

This implementation also maintains span counts on the express pointers, giving
O(log n) rank queries (ZRANK: "what position is this member?") and select-by-rank
(ZRANGE by index) — the operations a plain balanced BST cannot do without extra
bookkeeping.

Ordered by (score, member), like Redis. Zero dependencies.
"""

from __future__ import annotations
import random
from typing import Any, Iterator, List, Optional, Tuple

_MAX_LEVEL = 32
_P = 0.5


class _SkipNode:
    __slots__ = ("score", "member", "forward", "span")

    def __init__(self, score: float, member: Any, level: int):
        self.score = score
        self.member = member
        # forward[i] = next node at level i
        self.forward: List[Optional["_SkipNode"]] = [None] * level
        # span[i] = number of bottom-level nodes skipped by forward[i]
        self.span: List[int] = [0] * level


def _key(score: float, member: Any) -> Tuple[float, Any]:
    return (score, member)


class SkipList:
    """
    An ordered collection of (score, member) pairs, sorted by (score, member).

    Members are unique: re-adding an existing member updates its score
    (re-positioning it), exactly like Redis ZADD.
    """

    def __init__(self, seed: Optional[int] = None):
        self._head = _SkipNode(float("-inf"), None, _MAX_LEVEL)
        self._level = 1                  # current highest level in use
        self._size = 0
        self._rng = random.Random(seed)
        self._scores: dict = {}          # member -> score (for O(1) lookup)

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        lvl = 1
        while self._rng.random() < _P and lvl < _MAX_LEVEL:
            lvl += 1
        return lvl

    # ------------------------------------------------------------------
    # Insertion (ZADD)
    # ------------------------------------------------------------------

    def add(self, score: float, member: Any) -> None:
        """Insert `member` with `score`, or move it if it already exists."""
        if member in self._scores:
            self.remove(member)

        update: List[_SkipNode] = [self._head] * _MAX_LEVEL
        rank: List[int] = [0] * _MAX_LEVEL
        node = self._head

        for i in range(self._level - 1, -1, -1):
            rank[i] = 0 if i == self._level - 1 else rank[i + 1]
            while (node.forward[i] is not None
                   and _key(node.forward[i].score, node.forward[i].member)
                       < _key(score, member)):
                rank[i] += node.span[i]
                node = node.forward[i]
            update[i] = node

        lvl = self._random_level()
        if lvl > self._level:
            for i in range(self._level, lvl):
                rank[i] = 0
                update[i] = self._head
                update[i].span[i] = self._size
            self._level = lvl

        new_node = _SkipNode(score, member, lvl)
        for i in range(lvl):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
            # Split the span of update[i] across the new node
            new_node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1

        # Higher untouched levels gain one skipped node
        for i in range(lvl, self._level):
            update[i].span[i] += 1

        self._scores[member] = score
        self._size += 1

    # ------------------------------------------------------------------
    # Deletion (ZREM)
    # ------------------------------------------------------------------

    def remove(self, member: Any) -> bool:
        """Remove `member`. Returns True if it was present."""
        if member not in self._scores:
            return False
        score = self._scores[member]

        update: List[_SkipNode] = [self._head] * _MAX_LEVEL
        node = self._head
        for i in range(self._level - 1, -1, -1):
            while (node.forward[i] is not None
                   and _key(node.forward[i].score, node.forward[i].member)
                       < _key(score, member)):
                node = node.forward[i]
            update[i] = node

        target = node.forward[0]
        if target is None or target.member != member:
            return False

        for i in range(self._level):
            if update[i].forward[i] is target:
                update[i].span[i] += target.span[i] - 1
                update[i].forward[i] = target.forward[i]
            else:
                update[i].span[i] -= 1

        while self._level > 1 and self._head.forward[self._level - 1] is None:
            self._level -= 1

        del self._scores[member]
        self._size -= 1
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def score(self, member: Any) -> Optional[float]:
        """ZSCORE: the score of `member`, or None."""
        return self._scores.get(member)

    def __contains__(self, member: Any) -> bool:
        return member in self._scores

    def rank(self, member: Any) -> Optional[int]:
        """
        ZRANK: 0-based index of `member` in sorted order, or None.
        Runs in O(log n) using the span counts.
        """
        if member not in self._scores:
            return None
        score = self._scores[member]
        node = self._head
        r = 0
        for i in range(self._level - 1, -1, -1):
            while (node.forward[i] is not None
                   and _key(node.forward[i].score, node.forward[i].member)
                       <= _key(score, member)):
                r += node.span[i]
                node = node.forward[i]
        return r - 1

    def select(self, index: int) -> Optional[Tuple[float, Any]]:
        """
        ZRANGE by index: the (score, member) at 0-based `index`, or None.
        O(log n) via span-count traversal (Redis zslGetElementByRank).
        """
        if index < 0 or index >= self._size:
            return None
        rank_target = index + 1          # spans count in 1-based ranks
        node = self._head
        traversed = 0
        for i in range(self._level - 1, -1, -1):
            while (node.forward[i] is not None
                   and traversed + node.span[i] <= rank_target):
                traversed += node.span[i]
                node = node.forward[i]
        if traversed == rank_target and node is not self._head:
            return (node.score, node.member)
        return None

    def range_by_score(self, lo: float, hi: float) -> Iterator[Tuple[float, Any]]:
        """ZRANGEBYSCORE: yield (score, member) for lo ≤ score ≤ hi, in order."""
        node = self._head
        for i in range(self._level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].score < lo:
                node = node.forward[i]
        node = node.forward[0]
        while node is not None and node.score <= hi:
            yield (node.score, node.member)
            node = node.forward[0]

    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        node = self._head.forward[0]
        while node is not None:
            yield (node.score, node.member)
            node = node.forward[0]

    def __len__(self) -> int:
        return self._size

    @property
    def levels(self) -> int:
        return self._level

    def __repr__(self) -> str:
        return f"SkipList(size={self._size}, levels={self._level})"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_skiplist() -> dict:
    """
    Build a leaderboard of 5,000 players with random scores (the canonical
    Redis ZSET use case), then exercise rank, select-by-rank and range queries —
    all O(log n) — and confirm them against a sorted reference.
    """
    sl = SkipList(seed=42)
    rng = random.Random(7)

    players = {f"player{i}": rng.randint(0, 100_000) for i in range(5000)}
    for member, score in players.items():
        sl.add(score, member)

    ordered = sorted(players.items(), key=lambda kv: (kv[1], kv[0]))

    # ZRANK of a specific player matches its position in the sorted reference
    sample = "player1234"
    expected_rank = next(i for i, (m, _) in enumerate(ordered) if m == sample)
    rank_ok = sl.rank(sample) == expected_rank

    # ZRANGE by index: top-3 highest scores == last 3 of ascending order
    top3 = [sl.select(sl._size - 1 - i) for i in range(3)]
    expected_top3 = [(s, m) for m, s in ordered[-3:][::-1]]
    select_ok = top3 == expected_top3

    return {
        "players": len(sl),
        "levels": sl.levels,
        "zrank_ok": rank_ok,
        "zrange_by_index_ok": select_ok,
        "iteration_sorted": list(sl) == [(s, m) for m, s in ordered],
        "top3_scores": [s for s, _ in top3],
    }
