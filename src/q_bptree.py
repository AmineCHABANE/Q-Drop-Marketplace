"""
Q-BPTREE — B+ tree (the index structure inside every relational database)
Bayer & McCreight 1972 (B-tree); Comer, "The Ubiquitous B-Tree", 1979

The on-disk index that powers PostgreSQL, MySQL/InnoDB, SQLite, Oracle, SQL
Server — and the metadata indexes of most filesystems (NTFS, HFS+, ext4 htree).

A B+ tree keeps all values in the leaves, links the leaves in a list, and keeps
the tree balanced under insertion by splitting full nodes from the bottom up.
The result: O(log n) point lookups AND O(log n + k) ordered range scans, with a
high branching factor so the tree stays shallow (3–4 levels for billions of keys).

This is the *other* storage-engine paradigm next to the LSM-tree (see q_lsm.py):
B+ trees optimize reads and in-place updates; LSM-trees optimize writes.

Zero dependencies.
"""

from __future__ import annotations
import bisect
from typing import Any, Iterator, List, Optional, Tuple


class _Node:
    __slots__ = ("keys", "children", "values", "next_leaf", "is_leaf")

    def __init__(self, is_leaf: bool):
        self.is_leaf = is_leaf
        self.keys: List[Any] = []
        # internal nodes use `children`; leaves use `values` + `next_leaf`
        self.children: List["_Node"] = []
        self.values: List[Any] = []
        self.next_leaf: Optional["_Node"] = None


class BPlusTree:
    """
    A B+ tree mapping ordered keys to values.

    :param order: maximum number of children per internal node (branching
        factor). A node splits when it would exceed `order - 1` keys.
        Must be ≥ 3.
    """

    def __init__(self, order: int = 32):
        if order < 3:
            raise ValueError("order must be ≥ 3")
        self.order = order
        self._root = _Node(is_leaf=True)
        self._height = 1
        self._size = 0

    # ------------------------------------------------------------------
    # Point operations
    # ------------------------------------------------------------------

    def get(self, key: Any) -> Optional[Any]:
        """Return the value for `key`, or None if absent."""
        leaf = self._find_leaf(key)
        i = bisect.bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        return None

    def __contains__(self, key: Any) -> bool:
        leaf = self._find_leaf(key)
        i = bisect.bisect_left(leaf.keys, key)
        return i < len(leaf.keys) and leaf.keys[i] == key

    def insert(self, key: Any, value: Any) -> None:
        """Insert or update `key -> value`."""
        root = self._root
        split = self._insert(root, key, value)
        if split is not None:
            # Root split: create a new root one level up
            sep_key, right = split
            new_root = _Node(is_leaf=False)
            new_root.keys = [sep_key]
            new_root.children = [root, right]
            self._root = new_root
            self._height += 1

    def _insert(self, node: _Node, key: Any, value: Any
                ) -> Optional[Tuple[Any, _Node]]:
        """
        Insert into the subtree rooted at `node`. If the node splits, return
        (separator_key, new_right_node); otherwise None.
        """
        if node.is_leaf:
            i = bisect.bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.values[i] = value          # update existing
                return None
            node.keys.insert(i, key)
            node.values.insert(i, value)
            self._size += 1
            if len(node.keys) > self.order - 1:
                return self._split_leaf(node)
            return None

        # Internal node: descend into the correct child
        i = bisect.bisect_right(node.keys, key)
        split = self._insert(node.children[i], key, value)
        if split is None:
            return None
        sep_key, right = split
        node.keys.insert(i, sep_key)
        node.children.insert(i + 1, right)
        if len(node.keys) > self.order - 1:
            return self._split_internal(node)
        return None

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split_leaf(self, leaf: _Node) -> Tuple[Any, _Node]:
        mid = len(leaf.keys) // 2
        right = _Node(is_leaf=True)
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        # Maintain the leaf linked list for range scans
        right.next_leaf = leaf.next_leaf
        leaf.next_leaf = right
        # In a B+ tree the separator is the first key of the right leaf (copied up)
        return right.keys[0], right

    def _split_internal(self, node: _Node) -> Tuple[Any, _Node]:
        mid = len(node.keys) // 2
        sep_key = node.keys[mid]               # middle key moves UP, not copied
        right = _Node(is_leaf=False)
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return sep_key, right

    # ------------------------------------------------------------------
    # Navigation & scans
    # ------------------------------------------------------------------

    def _find_leaf(self, key: Any) -> _Node:
        node = self._root
        while not node.is_leaf:
            i = bisect.bisect_right(node.keys, key)
            node = node.children[i]
        return node

    def range_scan(self, lo: Any, hi: Any) -> Iterator[Tuple[Any, Any]]:
        """
        Yield (key, value) for all keys in [lo, hi], in ascending order, by
        walking the leaf linked list — the operation B+ trees exist for.
        """
        leaf = self._find_leaf(lo)
        while leaf is not None:
            for k, v in zip(leaf.keys, leaf.values):
                if k < lo:
                    continue
                if k > hi:
                    return
                yield k, v
            leaf = leaf.next_leaf

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield every (key, value) in ascending key order."""
        leaf = self._root
        while not leaf.is_leaf:
            leaf = leaf.children[0]
        while leaf is not None:
            yield from zip(leaf.keys, leaf.values)
            leaf = leaf.next_leaf

    def keys(self) -> Iterator[Any]:
        for k, _ in self.items():
            yield k

    def min_key(self) -> Optional[Any]:
        node = self._root
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0] if node.keys else None

    def max_key(self) -> Optional[Any]:
        node = self._root
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1] if node.keys else None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return self._height

    def __len__(self) -> int:
        return self._size

    def check_invariants(self) -> bool:
        """
        Verify the B+ tree invariants: every leaf at the same depth, keys
        sorted, and the leaf chain reproduces the sorted key order.
        """
        # All leaves at the same depth?
        depths: List[int] = []

        def walk(node: _Node, depth: int) -> None:
            if node.is_leaf:
                depths.append(depth)
            else:
                for c in node.children:
                    walk(c, depth + 1)

        walk(self._root, 0)
        if len(set(depths)) > 1:
            return False

        # Leaf chain is globally sorted and matches size?
        seen = list(self.keys())
        if seen != sorted(seen):
            return False
        return len(seen) == self._size

    def __repr__(self) -> str:
        return f"BPlusTree(order={self.order}, size={self._size}, height={self._height})"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_bptree() -> dict:
    """
    Insert 10,000 keys in shuffled order, then show:
      - the tree stays balanced and shallow,
      - point lookups are correct,
      - a range scan returns keys in sorted order via the leaf chain.
    """
    import random
    rng = random.Random(42)
    tree = BPlusTree(order=16)

    keys = list(range(10_000))
    rng.shuffle(keys)
    for k in keys:
        tree.insert(k, f"val:{k}")

    # Point lookups
    point_ok = all(tree.get(k) == f"val:{k}" for k in (0, 4999, 9999))
    missing_ok = tree.get(10_001) is None

    # Range scan [100, 110]
    scanned = list(tree.range_scan(100, 110))
    range_ok = [k for k, _ in scanned] == list(range(100, 111))

    return {
        "size": len(tree),
        "height": tree.height,
        "branching_order": tree.order,
        "point_lookup_ok": point_ok,
        "missing_key_ok": missing_ok,
        "range_scan_keys": [k for k, _ in scanned],
        "range_scan_sorted": range_ok,
        "invariants_hold": tree.check_invariants(),
    }
