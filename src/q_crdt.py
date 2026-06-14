"""
Q-CRDT — Conflict-free Replicated Data Types
Shapiro, Preguiça, Baquero & Zawirski, "Conflict-free Replicated Data Types",
INRIA RR-7687 / SSS 2011

The mathematics behind local-first and real-time collaborative software:
Figma, Linear, Notion, Apple Notes, Automerge, Yjs, Redis CRDTs, Riak.

A CRDT is a data type whose replicas can be updated independently and
concurrently, then merged automatically with NO coordination and NO conflicts.
The merge function forms a join-semilattice, so it is:
    • commutative   merge(a,b) = merge(b,a)
    • associative   merge(merge(a,b),c) = merge(a,merge(b,c))
    • idempotent    merge(a,a) = a
These three laws guarantee Strong Eventual Consistency: any two replicas that
have seen the same set of updates are in the same state, regardless of order.

Implemented here (all state-based / CvRDT):
    • VectorClock     — causality tracking (happens-before vs concurrent)
    • GCounter        — grow-only counter
    • PNCounter       — increment/decrement counter
    • LWWRegister     — last-writer-wins register
    • ORSet           — observed-remove set (add-wins on concurrency)
    • RGA             — replicated growable array (collaborative text/sequence)

Zero dependencies. Every merge is proven convergent by an order-shuffling test.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Vector clock — the causality primitive
# ---------------------------------------------------------------------------

class VectorClock:
    """
    Maps each replica id to a monotonically increasing counter.

    Lets us decide, for any two events, whether one *happened-before* the other
    or whether they are *concurrent* (the case CRDTs must resolve deterministically).
    """

    def __init__(self, clock: Optional[Dict[str, int]] = None):
        self.clock: Dict[str, int] = dict(clock or {})

    def tick(self, node: str) -> "VectorClock":
        self.clock[node] = self.clock.get(node, 0) + 1
        return self

    def get(self, node: str) -> int:
        return self.clock.get(node, 0)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Element-wise max — the lattice join."""
        nodes = set(self.clock) | set(other.clock)
        return VectorClock({n: max(self.get(n), other.get(n)) for n in nodes})

    def happens_before(self, other: "VectorClock") -> bool:
        """True if self → other (self causally precedes other)."""
        nodes = set(self.clock) | set(other.clock)
        leq = all(self.get(n) <= other.get(n) for n in nodes)
        lt = any(self.get(n) < other.get(n) for n in nodes)
        return leq and lt

    def concurrent_with(self, other: "VectorClock") -> bool:
        """True if neither happens-before the other (a genuine conflict)."""
        return not (self.happens_before(other)
                    or other.happens_before(self)
                    or self.clock == other.clock)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VectorClock) and self.clock == other.clock

    def __repr__(self) -> str:
        return f"VC({self.clock})"


# ---------------------------------------------------------------------------
# G-Counter — grow-only counter
# ---------------------------------------------------------------------------

class GCounter:
    """
    Increment-only counter. Each replica owns one slot it alone increments;
    the value is the sum of all slots; merge takes the element-wise max.

    Used for view counts, like counts, metrics that only go up.
    """

    def __init__(self, node: str):
        self.node = node
        self._counts: Dict[str, int] = {}

    def increment(self, amount: int = 1) -> "GCounter":
        if amount < 0:
            raise ValueError("GCounter cannot decrement")
        self._counts[self.node] = self._counts.get(self.node, 0) + amount
        return self

    @property
    def value(self) -> int:
        return sum(self._counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        merged = GCounter(self.node)
        for n in set(self._counts) | set(other._counts):
            merged._counts[n] = max(self._counts.get(n, 0),
                                    other._counts.get(n, 0))
        return merged

    def __repr__(self) -> str:
        return f"GCounter({self.value}, {self._counts})"


# ---------------------------------------------------------------------------
# PN-Counter — increment AND decrement
# ---------------------------------------------------------------------------

class PNCounter:
    """
    Two G-Counters: P accumulates increments, N accumulates decrements.
    value = P − N.  Supports both directions while staying convergent.

    Used for inventory, vote tallies, shopping-cart quantities.
    """

    def __init__(self, node: str):
        self.node = node
        self._p = GCounter(node)
        self._n = GCounter(node)

    def increment(self, amount: int = 1) -> "PNCounter":
        self._p.increment(amount)
        return self

    def decrement(self, amount: int = 1) -> "PNCounter":
        self._n.increment(amount)
        return self

    @property
    def value(self) -> int:
        return self._p.value - self._n.value

    def merge(self, other: "PNCounter") -> "PNCounter":
        merged = PNCounter(self.node)
        merged._p = self._p.merge(other._p)
        merged._n = self._n.merge(other._n)
        return merged

    def __repr__(self) -> str:
        return f"PNCounter({self.value})"


# ---------------------------------------------------------------------------
# LWW-Register — last-writer-wins
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Stamp:
    timestamp: float
    node: str   # tiebreaker for identical timestamps (total order)


class LWWRegister:
    """
    Single-value register where the write with the highest (timestamp, node)
    wins. Concurrent writes resolve deterministically by the tiebreaker.

    Used for user-profile fields, document titles, presence status.
    """

    def __init__(self, node: str, value: Any = None, timestamp: float = 0.0):
        self.node = node
        self.value = value
        self._stamp = _Stamp(timestamp, node)

    def set(self, value: Any, timestamp: float) -> "LWWRegister":
        stamp = _Stamp(timestamp, self.node)
        if stamp > self._stamp:
            self.value = value
            self._stamp = stamp
        return self

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        winner = self if self._stamp >= other._stamp else other
        out = LWWRegister(self.node, winner.value, winner._stamp.timestamp)
        out._stamp = _Stamp(winner._stamp.timestamp, winner._stamp.node)
        return out

    def __repr__(self) -> str:
        return f"LWW({self.value!r}@{self._stamp.timestamp})"


# ---------------------------------------------------------------------------
# OR-Set — observed-remove set (add wins on concurrency)
# ---------------------------------------------------------------------------

class ORSet:
    """
    A set supporting add and remove with the intuitive semantics that a
    concurrent add and remove resolves as *add wins*.

    Each add tags the element with a globally unique id. remove only erases the
    tags it has actually observed, so a concurrent add (with a fresh, unseen
    tag) survives. This is the model behind collaborative tag/label sets.
    """

    def __init__(self, node: str):
        self.node = node
        self._counter = itertools.count()
        # element → set of unique add-tags currently live
        self._adds: Dict[Hashable, Set[str]] = {}
        # tombstoned tags
        self._removed: Set[str] = set()

    def _fresh_tag(self) -> str:
        return f"{self.node}:{next(self._counter)}"

    def add(self, element: Hashable) -> "ORSet":
        self._adds.setdefault(element, set()).add(self._fresh_tag())
        return self

    def remove(self, element: Hashable) -> "ORSet":
        # Tombstone every tag we currently observe for this element
        for tag in self._adds.get(element, set()):
            self._removed.add(tag)
        return self

    def contains(self, element: Hashable) -> bool:
        live = self._adds.get(element, set()) - self._removed
        return len(live) > 0

    def values(self) -> Set[Hashable]:
        return {e for e in self._adds if self.contains(e)}

    def merge(self, other: "ORSet") -> "ORSet":
        merged = ORSet(self.node)
        for e in set(self._adds) | set(other._adds):
            merged._adds[e] = (self._adds.get(e, set())
                               | other._adds.get(e, set()))
        merged._removed = self._removed | other._removed
        return merged

    def __repr__(self) -> str:
        return f"ORSet({self.values()})"


# ---------------------------------------------------------------------------
# RGA — Replicated Growable Array (collaborative text / sequence)
# ---------------------------------------------------------------------------

@dataclass
class _RGANode:
    id: Tuple[int, str]            # (lamport_clock, node) — globally unique, ordered
    value: Any
    after: Optional[Tuple[int, str]]   # id of the element this was inserted after
    deleted: bool = False


class RGA:
    """
    Replicated Growable Array — the CRDT behind collaborative text editing.

    Every inserted element gets a unique, totally-ordered id (Lamport clock +
    node). Insertions name the element they follow; concurrent insertions at the
    same position are ordered deterministically by id (higher id first), so all
    replicas converge to the identical sequence. Deletes are tombstones.

    This is a simplified cousin of the algorithms in Yjs / Automakerge / RGA.
    """

    def __init__(self, node: str):
        self.node = node
        self._clock = 0
        self._nodes: Dict[Tuple[int, str], _RGANode] = {}
        self._order: List[Tuple[int, str]] = []   # cached materialized order

    def _next_id(self) -> Tuple[int, str]:
        self._clock += 1
        return (self._clock, self.node)

    def _observe_clock(self, ts: int) -> None:
        self._clock = max(self._clock, ts)

    def insert_after(self, after: Optional[Tuple[int, str]], value: Any
                     ) -> Tuple[int, str]:
        """Insert `value` immediately after element `after` (None = head)."""
        nid = self._next_id()
        node = _RGANode(id=nid, value=value, after=after)
        self._nodes[nid] = node
        self._rebuild()
        return nid

    def delete(self, nid: Tuple[int, str]) -> "RGA":
        if nid in self._nodes:
            self._nodes[nid].deleted = True
            self._rebuild()
        return self

    def _rebuild(self) -> None:
        """Materialize the linear order from the (after) relation + id tiebreak."""
        # children[after_id] = list of nodes inserted after that id
        children: Dict[Optional[Tuple[int, str]], List[_RGANode]] = {}
        for node in self._nodes.values():
            children.setdefault(node.after, []).append(node)
        # Concurrent inserts at the same anchor: higher id first (deterministic)
        for lst in children.values():
            lst.sort(key=lambda n: n.id, reverse=True)

        order: List[Tuple[int, str]] = []

        def walk(anchor: Optional[Tuple[int, str]]) -> None:
            for child in children.get(anchor, []):
                order.append(child.id)
                walk(child.id)

        walk(None)
        self._order = order

    def to_list(self) -> List[Any]:
        """Visible (non-deleted) sequence."""
        return [self._nodes[nid].value for nid in self._order
                if not self._nodes[nid].deleted]

    def to_string(self) -> str:
        return "".join(str(v) for v in self.to_list())

    def ids(self) -> List[Tuple[int, str]]:
        """Element ids in materialized order (including tombstones)."""
        return list(self._order)

    def merge(self, other: "RGA") -> "RGA":
        merged = RGA(self.node)
        merged._clock = max(self._clock, other._clock)
        for nid, node in itertools.chain(self._nodes.items(),
                                         other._nodes.items()):
            if nid not in merged._nodes:
                merged._nodes[nid] = _RGANode(node.id, node.value,
                                              node.after, node.deleted)
            else:
                # A delete on either side wins (monotone tombstone)
                merged._nodes[nid].deleted |= node.deleted
        merged._rebuild()
        return merged

    def __repr__(self) -> str:
        return f"RGA({self.to_list()!r})"


# ---------------------------------------------------------------------------
# Convergence verification — the lattice laws, checked empirically
# ---------------------------------------------------------------------------

def verify_convergence(replicas: List[Any]) -> bool:
    """
    Merge a list of CRDT replicas in EVERY possible pairwise order and confirm
    they all reach the identical state. This is the empirical proof of
    commutativity + associativity + idempotence.

    Works for any CRDT exposing .merge() and a comparable observable
    (value / values() / to_list()).
    """
    def observe(c: Any) -> Any:
        if hasattr(c, "value"):
            return c.value
        if hasattr(c, "values"):
            return frozenset(c.values())
        if hasattr(c, "to_list"):
            return tuple(c.to_list())
        raise TypeError("CRDT has no observable")

    results = []
    for perm in itertools.permutations(range(len(replicas))):
        acc = replicas[perm[0]]
        for idx in perm[1:]:
            acc = acc.merge(replicas[idx])
        # Idempotence: merging again changes nothing
        acc = acc.merge(acc)
        results.append(observe(acc))

    return all(r == results[0] for r in results)


def demonstrate_collaborative_edit() -> dict:
    """
    Two users edit the same document offline, then sync.

    Alice types "HI" ; Bob concurrently types "YO" at the same anchor.
    After exchanging state, BOTH replicas show the identical merged text —
    no central server, no locks, no lost writes.
    """
    alice = RGA("alice")
    bob = RGA("bob")

    # Shared starting point: both insert a common anchor char "_"
    anchor = alice.insert_after(None, "_")
    # Replicate the anchor to Bob by merging
    bob = bob.merge(alice)

    # Offline divergence — both insert after the same anchor
    alice.insert_after(anchor, "A")   # Alice's edit
    bob.insert_after(anchor, "B")     # Bob's concurrent edit

    # Sync in both directions
    alice_final = alice.merge(bob)
    bob_final = bob.merge(alice)

    return {
        "alice_sees": alice_final.to_string(),
        "bob_sees": bob_final.to_string(),
        "converged": alice_final.to_list() == bob_final.to_list(),
    }
