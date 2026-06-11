"""
Q-LSM: Log-Structured Merge-Tree (LSM-Tree) — Reference Implementation

The LSM-tree is the storage engine behind LevelDB, RocksDB, Cassandra, and HBase.
Understanding how it works is essential for anyone building databases or storage systems.

Core idea: writes go to a fast in-memory buffer (MemTable); when full, it's flushed
to disk as an immutable sorted file (SSTable). Reads merge results from memory + disk.
Periodic compaction merges SSTables to reclaim space and maintain read performance.

This implementation uses in-memory "files" (sorted lists) instead of real disk I/O,
so it runs entirely in Python with no dependencies. The data structures and algorithms
are faithful to the real thing.

Usage:
    from q_lsm import LSMTree

    db = LSMTree(memtable_limit=1000)
    db.put("key", "value")
    print(db.get("key"))     # "value"
    db.delete("key")
    print(db.get("key"))     # None

    # Scan a range
    for key, value in db.scan("a", "z"):
        print(key, value)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


# Sentinel value indicating a deleted key (tombstone)
_TOMBSTONE = object()


# ---------------------------------------------------------------------------
# WAL (Write-Ahead Log)
# ---------------------------------------------------------------------------

class WAL:
    """
    Write-Ahead Log: every mutation is recorded here before the MemTable.
    On crash, replay WAL to reconstruct the MemTable.

    In this reference implementation the WAL is just an in-memory list.
    In a real system it's an append-only file (e.g. LevelDB's .log format).
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, object]] = []

    def append(self, key: str, value: object) -> None:
        self._entries.append((key, value))

    def replay(self) -> Iterator[Tuple[str, object]]:
        yield from self._entries

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# MemTable
# ---------------------------------------------------------------------------

class MemTable:
    """
    In-memory sorted write buffer.

    In a real LSM-tree this is typically a skip list or a red-black tree for
    O(log n) insert and lookup. Here we use a sorted list (bisect module)
    which is simpler to read but has O(n) insert — fine for education.

    When size exceeds the limit, the MemTable is flushed to a new SSTable
    and replaced with a fresh empty one.
    """

    def __init__(self) -> None:
        # Sorted list of (key, value) pairs. Keys are unique.
        self._data: List[Tuple[str, object]] = []
        self._size: int = 0  # number of entries (including tombstones)

    def put(self, key: str, value: object) -> None:
        """Insert or update key. O(log n) search + O(n) insert."""
        keys = [k for k, _ in self._data]
        idx = bisect.bisect_left(keys, key)
        if idx < len(self._data) and self._data[idx][0] == key:
            self._data[idx] = (key, value)  # update in place
        else:
            self._data.insert(idx, (key, value))
            self._size += 1

    def get(self, key: str) -> Optional[object]:
        """O(log n) lookup. Returns _TOMBSTONE if deleted, None if absent."""
        keys = [k for k, _ in self._data]
        idx = bisect.bisect_left(keys, key)
        if idx < len(self._data) and self._data[idx][0] == key:
            return self._data[idx][1]
        return None

    def delete(self, key: str) -> None:
        """Write a tombstone marker for key."""
        self.put(key, _TOMBSTONE)

    def to_sorted_pairs(self) -> List[Tuple[str, object]]:
        """Return all entries sorted by key (for SSTable flush)."""
        return list(self._data)

    def __len__(self) -> int:
        return self._size

    def scan(self, start: str, end: str) -> Iterator[Tuple[str, object]]:
        """Yield (key, value) pairs where start <= key <= end."""
        keys = [k for k, _ in self._data]
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_right(keys, end)
        for i in range(lo, hi):
            yield self._data[i]


# ---------------------------------------------------------------------------
# SSTable
# ---------------------------------------------------------------------------

@dataclass
class SSTable:
    """
    Sorted String Table: immutable, sorted sequence of key-value pairs.

    In a real system this is a file on disk with:
    - Data blocks (compressed key-value pairs)
    - Index block (one entry per data block for fast lookup)
    - Bloom filter (to skip tables that don't contain the key)
    - Footer (offsets of index and meta blocks)

    Here we store everything in a sorted Python list (same logical structure).
    """

    level: int
    entries: List[Tuple[str, object]] = field(default_factory=list)
    _seq: int = field(default=0)  # sequence number for ordering within level

    def get(self, key: str) -> Optional[object]:
        """Binary search. O(log n)."""
        keys = [k for k, _ in self.entries]
        idx = bisect.bisect_left(keys, key)
        if idx < len(self.entries) and self.entries[idx][0] == key:
            return self.entries[idx][1]
        return None

    def scan(self, start: str, end: str) -> Iterator[Tuple[str, object]]:
        """Yield pairs in [start, end] range."""
        keys = [k for k, _ in self.entries]
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_right(keys, end)
        for i in range(lo, hi):
            yield self.entries[i]

    @property
    def min_key(self) -> Optional[str]:
        return self.entries[0][0] if self.entries else None

    @property
    def max_key(self) -> Optional[str]:
        return self.entries[-1][0] if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def merge_sstables(tables: List[SSTable], target_level: int) -> SSTable:
    """
    Merge multiple SSTables into one, keeping only the latest value per key.

    This is the core of LSM compaction. Real implementations:
    - LevelDB: merge L0 (overlapping) → L1 (non-overlapping), then Ln → Ln+1
    - RocksDB: Leveled or Tiered compaction strategies

    Algorithm: k-way merge (like merge sort) over sorted iterators.
    For duplicate keys, prefer the entry from the newer table (lower index here).
    """
    # Convert each table to an iterator of (key, value)
    iters = [iter(t.entries) for t in tables]
    heads: List[Optional[Tuple[str, object]]] = [next(it, None) for it in iters]

    merged: List[Tuple[str, object]] = []
    last_key: Optional[str] = None

    while any(h is not None for h in heads):
        # Find iterator with the lexicographically smallest key
        # (when keys are equal, pick the first iterator = newest table)
        min_idx = None
        min_key = None
        for i, h in enumerate(heads):
            if h is None:
                continue
            if min_key is None or h[0] < min_key:
                min_key = h[0]
                min_idx = i

        if min_idx is None:
            break

        key, value = heads[min_idx]
        heads[min_idx] = next(iters[min_idx], None)

        # Skip duplicate keys (they come from older tables — already superseded)
        if key == last_key:
            continue
        last_key = key

        # Skip tombstones in the output (they've served their purpose)
        if value is not _TOMBSTONE:
            merged.append((key, value))

    return SSTable(level=target_level, entries=merged)


# ---------------------------------------------------------------------------
# LSM Tree
# ---------------------------------------------------------------------------

class LSMTree:
    """
    Log-Structured Merge-Tree.

    Write path:
        put(k, v)
          → append to WAL
          → insert into MemTable
          → if MemTable full: flush to L0 SSTable, clear MemTable + WAL
          → if L0 has too many SSTables: compact L0 → L1

    Read path:
        get(k)
          → check MemTable (newest)
          → check L0 SSTables newest→oldest (they can overlap)
          → check L1, L2, ... SSTables (non-overlapping per level)

    This implements a simplified two-level compaction (L0 + L1).
    Real systems have many levels (LevelDB: up to 7, each 10x larger).
    """

    def __init__(
        self,
        memtable_limit: int = 1000,
        l0_compaction_trigger: int = 4,
    ) -> None:
        """
        memtable_limit        : flush MemTable to L0 when it reaches this size
        l0_compaction_trigger : compact all L0 SSTables to L1 when L0 has this many
        """
        self.memtable_limit = memtable_limit
        self.l0_trigger = l0_compaction_trigger

        self._wal = WAL()
        self._memtable = MemTable()

        # levels[0] = L0 SSTables (may overlap), levels[1] = L1 (sorted, no overlap)
        self._levels: List[List[SSTable]] = [[], []]
        self._sstable_seq: int = 0

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def put(self, key: str, value: str) -> None:
        """Insert or update a key."""
        self._wal.append(key, value)
        self._memtable.put(key, value)
        self._maybe_flush()

    def delete(self, key: str) -> None:
        """Mark key as deleted (tombstone)."""
        self._wal.append(key, _TOMBSTONE)
        self._memtable.delete(key)
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Flush MemTable to L0 if size limit reached."""
        if len(self._memtable) >= self.memtable_limit:
            self._flush_memtable()
            if len(self._levels[0]) >= self.l0_trigger:
                self._compact_l0_to_l1()

    def _flush_memtable(self) -> None:
        """Write MemTable to a new L0 SSTable."""
        entries = self._memtable.to_sorted_pairs()
        if not entries:
            return
        self._sstable_seq += 1
        sst = SSTable(level=0, entries=entries, _seq=self._sstable_seq)
        self._levels[0].append(sst)
        self._memtable = MemTable()
        self._wal.clear()

    def _compact_l0_to_l1(self) -> None:
        """
        Compact all L0 SSTables + existing L1 into a new L1.

        L0 tables overlap each other (writes come in any order),
        so we must merge ALL of them. L1 is already sorted/non-overlapping.
        """
        all_tables = self._levels[0] + self._levels[1]
        # Sort by sequence number so newer entries win during merge
        all_tables.sort(key=lambda t: t._seq, reverse=True)
        merged = merge_sstables(all_tables, target_level=1)
        self._levels[0] = []
        self._levels[1] = [merged]

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """
        Look up a key.

        Search order (newest → oldest):
          1. MemTable
          2. L0 SSTables (newest first — they may overlap)
          3. L1 SSTables
        """
        # 1. MemTable
        val = self._memtable.get(key)
        if val is not None:
            return None if val is _TOMBSTONE else val  # type: ignore

        # 2. L0 (newest SSTable first)
        for sst in reversed(self._levels[0]):
            val = sst.get(key)
            if val is not None:
                return None if val is _TOMBSTONE else val  # type: ignore

        # 3. L1
        for sst in self._levels[1]:
            val = sst.get(key)
            if val is not None:
                return None if val is _TOMBSTONE else val  # type: ignore

        return None

    def scan(self, start: str, end: str) -> Iterator[Tuple[str, str]]:
        """
        Range scan returning (key, value) pairs where start <= key <= end.

        Merges results from MemTable + all SSTables, deduplicates by key
        (newest version wins), and strips tombstones.
        """
        # Collect all sources
        sources: List[Iterator[Tuple[str, object]]] = [
            self._memtable.scan(start, end)
        ]
        for sst in reversed(self._levels[0]):
            sources.append(sst.scan(start, end))
        for sst in self._levels[1]:
            sources.append(sst.scan(start, end))

        # Heap-merge (same logic as compaction merge)
        # We tag each entry with source index so newer source wins on tie
        seen: set = set()
        # Collect all then sort: simple but not streaming. Good enough for reference.
        all_entries: List[Tuple[str, int, object]] = []
        for src_idx, src in enumerate(sources):
            for key, value in src:
                all_entries.append((key, src_idx, value))

        # Sort by key, then by source_idx ascending (source 0 = MemTable = newest)
        all_entries.sort(key=lambda x: (x[0], x[1]))

        for key, _src_idx, value in all_entries:
            if key in seen:
                continue
            seen.add(key)
            if value is not _TOMBSTONE:
                yield key, value  # type: ignore

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self) -> None:
        """Manual compaction trigger (flush + merge all levels)."""
        self._flush_memtable()
        self._compact_l0_to_l1()

    def stats(self) -> dict:
        """Return tree statistics."""
        return {
            "memtable_size": len(self._memtable),
            "wal_entries": len(self._wal),
            "l0_sstables": len(self._levels[0]),
            "l0_entries": sum(len(s) for s in self._levels[0]),
            "l1_sstables": len(self._levels[1]),
            "l1_entries": sum(len(s) for s in self._levels[1]),
            "total_entries": (
                len(self._memtable)
                + sum(len(s) for s in self._levels[0])
                + sum(len(s) for s in self._levels[1])
            ),
        }
