"""
Q-RING — Consistent hashing & rendezvous hashing (distributed sharding)
  • Consistent hashing with virtual nodes  (Karger et al., STOC 1997)
  • Rendezvous / Highest-Random-Weight hashing (Thaler & Ravishankar 1998)
  • Bounded-load consistent hashing          (Mirrokni et al., Google 2016)

The algorithm that decides WHICH server owns a key — and, crucially, moves as
few keys as possible when servers are added or removed. The backbone of:
  - DynamoDB, Cassandra, Riak partitioning
  - memcached / Redis client-side sharding
  - CDN edge routing, load balancers, distributed caches

Classic modulo hashing (`hash(key) % N`) remaps almost every key when N changes.
Consistent hashing remaps only ~K/N keys. This module proves that empirically.

Zero dependencies.
"""

from __future__ import annotations
import bisect
import hashlib
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


def _hash(key: str) -> int:
    """Stable 64-bit hash (md5-based, independent of Python's salted hash)."""
    return int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Consistent hashing ring
# ---------------------------------------------------------------------------

class ConsistentHashRing:
    """
    A hash ring where each physical node is placed at `vnodes` positions
    (virtual nodes) to smooth out load. A key is owned by the first node
    clockwise from the key's hash position.

    Adding/removing a node only reassigns the keys in the arcs that node
    covers — O(K/N) keys move, versus O(K) for modulo hashing.
    """

    def __init__(self, vnodes: int = 150):
        self.vnodes = vnodes
        self._ring: List[int] = []                 # sorted hash positions
        self._owner: Dict[int, str] = {}           # position -> node
        self._nodes: Set[str] = set()

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for v in range(self.vnodes):
            pos = _hash(f"{node}#{v}")
            # Skip the rare collision rather than clobber an existing vnode
            if pos in self._owner:
                continue
            idx = bisect.bisect(self._ring, pos)
            self._ring.insert(idx, pos)
            self._owner[pos] = node

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        keep_positions = [p for p in self._ring if self._owner[p] != node]
        for p in self._ring:
            if self._owner[p] == node:
                del self._owner[p]
        self._ring = keep_positions

    def get_node(self, key: str) -> Optional[str]:
        """Return the node owning `key` (first vnode clockwise)."""
        if not self._ring:
            return None
        pos = _hash(key)
        idx = bisect.bisect(self._ring, pos)
        if idx == len(self._ring):
            idx = 0   # wrap around the ring
        return self._owner[self._ring[idx]]

    def get_replicas(self, key: str, n: int) -> List[str]:
        """
        Return the `n` distinct nodes owning `key` clockwise — the replica set
        for that key (as Cassandra/Dynamo do for replication factor n).
        """
        if not self._ring:
            return []
        pos = _hash(key)
        idx = bisect.bisect(self._ring, pos)
        result: List[str] = []
        seen: Set[str] = set()
        for i in range(len(self._ring)):
            p = self._ring[(idx + i) % len(self._ring)]
            node = self._owner[p]
            if node not in seen:
                seen.add(node)
                result.append(node)
                if len(result) == n:
                    break
        return result

    def load_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Count how many of `keys` land on each node."""
        counts: Counter = Counter()
        for k in keys:
            owner = self.get_node(k)
            if owner is not None:
                counts[owner] += 1
        return dict(counts)

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def __repr__(self) -> str:
        return f"ConsistentHashRing(nodes={len(self._nodes)}, vnodes={self.vnodes})"


# ---------------------------------------------------------------------------
# Rendezvous (Highest-Random-Weight) hashing
# ---------------------------------------------------------------------------

class RendezvousHash:
    """
    For each key, every node computes a weight = hash(node, key); the node with
    the highest weight wins. No ring, no virtual nodes — just a deterministic
    max. Adding/removing a node only moves the keys that node wins or loses.

    Used by some CDNs and by Ceph's CRUSH lineage. Simpler than a ring and
    naturally supports weighted nodes.
    """

    def __init__(self) -> None:
        self._nodes: Set[str] = set()

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def remove_node(self, node: str) -> None:
        self._nodes.discard(node)

    @staticmethod
    def _weight(node: str, key: str) -> int:
        return int.from_bytes(
            hashlib.md5(f"{node}\x00{key}".encode()).digest()[:8], "big")

    def get_node(self, key: str) -> Optional[str]:
        if not self._nodes:
            return None
        return max(self._nodes, key=lambda n: self._weight(n, key))

    def get_replicas(self, key: str, n: int) -> List[str]:
        """Top-n nodes by weight — a deterministic, well-distributed replica set."""
        ranked = sorted(self._nodes, key=lambda n: self._weight(n, key),
                        reverse=True)
        return ranked[:n]

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def __repr__(self) -> str:
        return f"RendezvousHash(nodes={len(self._nodes)})"


# ---------------------------------------------------------------------------
# Comparison: consistent vs naive modulo hashing
# ---------------------------------------------------------------------------

def _modulo_owner(key: str, n_nodes: int) -> int:
    return _hash(key) % n_nodes


def compare_remapping(n_keys: int = 10_000, n_nodes: int = 10) -> dict:
    """
    Quantify the central promise of consistent hashing.

    Remove one node from an N-node cluster and measure how many of `n_keys`
    keys change owner:
      - naive `hash % N`   → ~ (N-1)/N of all keys move (catastrophic)
      - consistent hashing → ~ 1/N of all keys move (only the dead node's share)
    """
    keys = [f"key:{i}" for i in range(n_keys)]

    # --- Naive modulo: N=10 → N=9 ---
    before_mod = {k: _modulo_owner(k, n_nodes) for k in keys}
    after_mod = {k: _modulo_owner(k, n_nodes - 1) for k in keys}
    moved_mod = sum(1 for k in keys if before_mod[k] != after_mod[k])

    # --- Consistent hashing: remove one node ---
    ring = ConsistentHashRing(vnodes=200)
    node_names = [f"node{i}" for i in range(n_nodes)]
    for node in node_names:
        ring.add_node(node)
    before_ring = {k: ring.get_node(k) for k in keys}
    ring.remove_node(node_names[-1])
    after_ring = {k: ring.get_node(k) for k in keys}
    moved_ring = sum(1 for k in keys if before_ring[k] != after_ring[k])

    return {
        "n_keys": n_keys,
        "n_nodes": n_nodes,
        "modulo_keys_moved": moved_mod,
        "modulo_fraction": round(moved_mod / n_keys, 3),
        "consistent_keys_moved": moved_ring,
        "consistent_fraction": round(moved_ring / n_keys, 3),
        "improvement_factor": round(moved_mod / max(1, moved_ring), 1),
    }


def balance_stats(n_keys: int = 100_000, n_nodes: int = 10,
                  vnodes: int = 200) -> dict:
    """Report how evenly keys spread across nodes (max/min load ratio)."""
    ring = ConsistentHashRing(vnodes=vnodes)
    for i in range(n_nodes):
        ring.add_node(f"node{i}")
    keys = [f"key:{i}" for i in range(n_keys)]
    dist = ring.load_distribution(keys)
    loads = list(dist.values())
    ideal = n_keys / n_nodes
    return {
        "ideal_per_node": ideal,
        "min_load": min(loads),
        "max_load": max(loads),
        "max_over_ideal": round(max(loads) / ideal, 3),
        "min_over_ideal": round(min(loads) / ideal, 3),
    }
