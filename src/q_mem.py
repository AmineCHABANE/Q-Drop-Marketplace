"""
Q-MEM: Arena Allocator — Reference Implementation

Arena (a.k.a. region/pool) allocation is the memory management strategy behind
jemalloc's arenas, Apache's APR pools, Protobuf arenas, and most game engines.

Core idea: instead of tracking each allocation individually (like malloc/free),
allocate from large contiguous chunks ("slabs") and free everything at once.
This gives:
- O(1) allocation (bump a pointer)
- Zero per-object free overhead (free the whole arena)
- No fragmentation within an arena
- Better cache locality (objects allocated together live together)

This Python implementation simulates the memory layout with bytearrays so you
can see exactly how size classes, slabs, free lists, and fragmentation work.
The algorithms are the real thing — only the "memory" is simulated.

Usage:
    from q_mem import ArenaAllocator

    arena = ArenaAllocator(slab_size=4096)
    ptr1 = arena.alloc(100)        # returns an Allocation handle
    ptr2 = arena.alloc(250)
    arena.write(ptr1, b"hello")
    print(arena.read(ptr1, 5))     # b"hello"
    arena.free(ptr1)               # returns block to free list
    print(arena.stats())
    arena.reset()                  # free EVERYTHING in O(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Size classes
# ---------------------------------------------------------------------------

def build_size_classes(max_size: int = 4096) -> List[int]:
    """
    Build jemalloc-style size classes.

    Small allocations are rounded up to the nearest size class to limit
    the number of distinct free lists while keeping internal fragmentation
    bounded (~25% worst case with 4 classes per doubling).

    Classes: 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, ...
    (linear steps within each power-of-two group, like jemalloc)
    """
    classes = [8, 16, 24, 32, 40, 48, 56, 64]
    base = 64
    while base < max_size:
        step = base // 4
        for i in range(1, 5):
            size = base + step * i
            if size <= max_size:
                classes.append(size)
        base *= 2
    return classes


def size_class_for(size: int, classes: List[int]) -> Optional[int]:
    """Round a requested size up to its size class (binary search)."""
    lo, hi = 0, len(classes)
    while lo < hi:
        mid = (lo + hi) // 2
        if classes[mid] < size:
            lo = mid + 1
        else:
            hi = mid
    return classes[lo] if lo < len(classes) else None


# ---------------------------------------------------------------------------
# Slab
# ---------------------------------------------------------------------------

@dataclass
class Slab:
    """
    A contiguous chunk of memory carved into equal-size blocks.

    Real allocators mmap() slabs from the OS (often 2MB huge pages).
    Here we simulate with a bytearray.

    Layout for block_size=64, slab_size=4096:
        [block 0][block 1][block 2]...[block 63]

    A free list (stack of block indices) tracks available blocks.
    Allocation = pop from free list. Free = push back. Both O(1).
    """
    slab_id: int
    block_size: int
    slab_size: int
    memory: bytearray = field(init=False)
    free_blocks: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.memory = bytearray(self.slab_size)
        num_blocks = self.slab_size // self.block_size
        # LIFO free list: reusing recently-freed blocks is cache-friendly
        self.free_blocks = list(range(num_blocks - 1, -1, -1))

    @property
    def num_blocks(self) -> int:
        return self.slab_size // self.block_size

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def is_full(self) -> bool:
        return not self.free_blocks

    @property
    def is_empty(self) -> bool:
        return len(self.free_blocks) == self.num_blocks

    def alloc_block(self) -> Optional[int]:
        """Pop a free block index. O(1)."""
        return self.free_blocks.pop() if self.free_blocks else None

    def free_block(self, block_idx: int) -> None:
        """Push a block back onto the free list. O(1)."""
        self.free_blocks.append(block_idx)


# ---------------------------------------------------------------------------
# Allocation handle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Allocation:
    """
    Handle to an allocated block (the 'pointer' returned by alloc).

    requested_size vs block_size difference = internal fragmentation.
    """
    slab_id: int
    block_idx: int
    offset: int          # byte offset within the slab
    requested_size: int
    block_size: int

    @property
    def internal_fragmentation(self) -> int:
        return self.block_size - self.requested_size


# ---------------------------------------------------------------------------
# Arena Allocator
# ---------------------------------------------------------------------------

class ArenaAllocator:
    """
    Slab-based arena allocator with size classes.

    alloc(size):
        1. Round size up to nearest size class
        2. Find a non-full slab for that class (or create one)
        3. Pop a block from its free list → O(1)

    free(allocation):
        1. Push block back to its slab's free list → O(1)
        2. (Optionally) release fully-empty slabs back to the OS

    reset():
        Drop all slabs at once → O(1) per slab. This is the arena superpower:
        perfect for request-scoped or frame-scoped memory (web servers, games).

    Allocations larger than the biggest size class get a dedicated
    "large object" slab, like jemalloc's large/huge classes.
    """

    def __init__(self, slab_size: int = 4096) -> None:
        self.slab_size = slab_size
        self.size_classes = build_size_classes(max_size=slab_size // 4)

        # size_class → list of slabs serving that class
        self._slabs_by_class: Dict[int, List[Slab]] = {}
        self._slabs_by_id: Dict[int, Slab] = {}
        self._next_slab_id: int = 0

        # Counters for stats
        self._total_allocs: int = 0
        self._total_frees: int = 0
        self._live_allocs: int = 0
        self._bytes_requested: int = 0
        self._bytes_allocated: int = 0  # after size-class rounding

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def alloc(self, size: int) -> Allocation:
        """Allocate `size` bytes. Returns an Allocation handle."""
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        block_size = size_class_for(size, self.size_classes)
        if block_size is None:
            # Large object: dedicated slab with a single block
            block_size = size
            slab = self._new_slab(block_size, slab_size=size)
        else:
            slab = self._find_or_create_slab(block_size)

        block_idx = slab.alloc_block()
        assert block_idx is not None  # _find_or_create_slab guarantees space

        self._total_allocs += 1
        self._live_allocs += 1
        self._bytes_requested += size
        self._bytes_allocated += block_size

        return Allocation(
            slab_id=slab.slab_id,
            block_idx=block_idx,
            offset=block_idx * slab.block_size,
            requested_size=size,
            block_size=block_size,
        )

    def _find_or_create_slab(self, block_size: int) -> Slab:
        """Find a slab with free space for this size class, or create one."""
        slabs = self._slabs_by_class.setdefault(block_size, [])
        for slab in slabs:
            if not slab.is_full:
                return slab
        return self._new_slab(block_size, slab_size=self.slab_size)

    def _new_slab(self, block_size: int, slab_size: int) -> Slab:
        slab = Slab(
            slab_id=self._next_slab_id,
            block_size=block_size,
            slab_size=slab_size,
        )
        self._next_slab_id += 1
        self._slabs_by_id[slab.slab_id] = slab
        self._slabs_by_class.setdefault(block_size, []).append(slab)
        return slab

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, allocation: Allocation) -> None:
        """Return a block to its slab's free list. O(1)."""
        slab = self._slabs_by_id.get(allocation.slab_id)
        if slab is None:
            raise ValueError(f"Unknown slab {allocation.slab_id} (double free after reset?)")

        slab.free_block(allocation.block_idx)
        self._total_frees += 1
        self._live_allocs -= 1
        self._bytes_requested -= allocation.requested_size
        self._bytes_allocated -= allocation.block_size

        # Release fully-empty slabs (like jemalloc's purging)
        if slab.is_empty:
            self._release_slab(slab)

    def _release_slab(self, slab: Slab) -> None:
        del self._slabs_by_id[slab.slab_id]
        self._slabs_by_class[slab.block_size].remove(slab)

    def reset(self) -> None:
        """Free ALL allocations at once. The O(1)-per-slab arena superpower."""
        self._slabs_by_class.clear()
        self._slabs_by_id.clear()
        self._live_allocs = 0
        self._bytes_requested = 0
        self._bytes_allocated = 0

    # ------------------------------------------------------------------
    # Memory access (simulated)
    # ------------------------------------------------------------------

    def write(self, allocation: Allocation, data: bytes) -> None:
        """Write bytes into an allocated block (bounds-checked)."""
        if len(data) > allocation.block_size:
            raise ValueError(
                f"Write of {len(data)} bytes exceeds block size {allocation.block_size}"
            )
        slab = self._slabs_by_id[allocation.slab_id]
        start = allocation.offset
        slab.memory[start:start + len(data)] = data

    def read(self, allocation: Allocation, length: int) -> bytes:
        """Read bytes from an allocated block (bounds-checked)."""
        if length > allocation.block_size:
            raise ValueError(
                f"Read of {length} bytes exceeds block size {allocation.block_size}"
            )
        slab = self._slabs_by_id[allocation.slab_id]
        start = allocation.offset
        return bytes(slab.memory[start:start + length])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Allocator statistics, including fragmentation analysis."""
        total_slab_bytes = sum(s.slab_size for s in self._slabs_by_id.values())
        internal_frag = self._bytes_allocated - self._bytes_requested
        return {
            "live_allocations": self._live_allocs,
            "total_allocs": self._total_allocs,
            "total_frees": self._total_frees,
            "num_slabs": len(self._slabs_by_id),
            "slab_bytes": total_slab_bytes,
            "bytes_requested": self._bytes_requested,
            "bytes_allocated": self._bytes_allocated,
            "internal_fragmentation_bytes": internal_frag,
            "internal_fragmentation_pct": (
                100.0 * internal_frag / self._bytes_allocated
                if self._bytes_allocated else 0.0
            ),
            "num_size_classes": len(self.size_classes),
        }
