"""
Test suite for all Q-Drop reference implementations.

Run from the repository root:
    python -m unittest discover -s src/tests -v
or:
    cd src && python -m tests.test_all

No external dependencies — pure standard library.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from q_hnsw import HNSWGraph
from q_lsm import LSMTree
from q_col import ColumnStore, DictColumn, RLEColumn, Mask
from q_stream import StreamProcessor, SessionWindowProcessor, Event
from q_mem import ArenaAllocator, build_size_classes, size_class_for


# ---------------------------------------------------------------------------
# Q-HNSW
# ---------------------------------------------------------------------------

class TestHNSW(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_insert_and_exact_match(self):
        index = HNSWGraph(dim=4, M=8)
        vectors = {i: [random.random() for _ in range(4)] for i in range(50)}
        for nid, vec in vectors.items():
            index.insert(nid, vec)

        # Searching for an indexed vector should return it as the top result
        results = index.search(vectors[7], k=1, ef=50)
        self.assertEqual(results[0][1], 7)
        self.assertAlmostEqual(results[0][0], 0.0, places=6)

    def test_recall_against_brute_force(self):
        """HNSW results should substantially overlap exact nearest neighbors."""
        dim, n = 8, 200
        index = HNSWGraph(dim=dim, M=16, ef_construction=100)
        vectors = {i: [random.gauss(0, 1) for _ in range(dim)] for i in range(n)}
        for nid, vec in vectors.items():
            index.insert(nid, vec)

        query = [random.gauss(0, 1) for _ in range(dim)]
        k = 10

        # Brute-force ground truth
        exact = sorted(
            (HNSWGraph._cosine_dist(query, v), nid) for nid, v in vectors.items()
        )[:k]
        exact_ids = {nid for _, nid in exact}

        approx = index.search(query, k=k, ef=100)
        approx_ids = {nid for _, nid in approx}

        recall = len(exact_ids & approx_ids) / k
        self.assertGreaterEqual(recall, 0.8, f"Recall too low: {recall}")

    def test_l2_distance(self):
        index = HNSWGraph(dim=2, M=4, distance="l2")
        index.insert(0, [0.0, 0.0])
        index.insert(1, [10.0, 10.0])
        index.insert(2, [1.0, 1.0])
        results = index.search([0.5, 0.5], k=2)
        self.assertIn(results[0][1], (0, 2))

    def test_empty_and_len(self):
        index = HNSWGraph(dim=3)
        self.assertEqual(index.search([1, 2, 3], k=5), [])
        self.assertEqual(len(index), 0)
        index.insert(0, [1, 2, 3])
        self.assertEqual(len(index), 1)

    def test_dim_mismatch_raises(self):
        index = HNSWGraph(dim=3)
        with self.assertRaises(ValueError):
            index.insert(0, [1, 2])

    def test_stats(self):
        index = HNSWGraph(dim=2, M=4)
        for i in range(20):
            index.insert(i, [random.random(), random.random()])
        s = index.stats()
        self.assertEqual(s["num_vectors"], 20)
        self.assertGreater(s["total_edges"], 0)


# ---------------------------------------------------------------------------
# Q-LSM
# ---------------------------------------------------------------------------

class TestLSM(unittest.TestCase):
    def test_put_get(self):
        db = LSMTree(memtable_limit=100)
        db.put("a", "1")
        db.put("b", "2")
        self.assertEqual(db.get("a"), "1")
        self.assertEqual(db.get("b"), "2")
        self.assertIsNone(db.get("missing"))

    def test_update_overwrites(self):
        db = LSMTree(memtable_limit=100)
        db.put("k", "v1")
        db.put("k", "v2")
        self.assertEqual(db.get("k"), "v2")

    def test_delete_tombstone(self):
        db = LSMTree(memtable_limit=100)
        db.put("k", "v")
        db.delete("k")
        self.assertIsNone(db.get("k"))

    def test_flush_and_read_from_sstable(self):
        """Values must survive a MemTable flush to L0."""
        db = LSMTree(memtable_limit=10)
        for i in range(25):  # forces 2 flushes
            db.put(f"key{i:03d}", f"val{i}")
        self.assertGreaterEqual(db.stats()["l0_sstables"], 1)
        for i in range(25):
            self.assertEqual(db.get(f"key{i:03d}"), f"val{i}")

    def test_compaction_preserves_data(self):
        db = LSMTree(memtable_limit=10, l0_compaction_trigger=2)
        for i in range(100):
            db.put(f"key{i:03d}", f"val{i}")
        db.compact()
        self.assertEqual(db.stats()["l0_sstables"], 0)
        for i in range(100):
            self.assertEqual(db.get(f"key{i:03d}"), f"val{i}")

    def test_delete_survives_compaction(self):
        db = LSMTree(memtable_limit=5, l0_compaction_trigger=2)
        for i in range(20):
            db.put(f"key{i:02d}", f"val{i}")
        db.delete("key05")
        db.compact()
        self.assertIsNone(db.get("key05"))
        self.assertEqual(db.get("key06"), "val6")

    def test_newest_value_wins_after_compaction(self):
        db = LSMTree(memtable_limit=5, l0_compaction_trigger=2)
        for round_num in range(5):
            for i in range(10):
                db.put(f"key{i}", f"round{round_num}")
        db.compact()
        for i in range(10):
            self.assertEqual(db.get(f"key{i}"), "round4")

    def test_scan_range(self):
        db = LSMTree(memtable_limit=5)
        for c in "abcdefgh":
            db.put(c, c.upper())
        result = dict(db.scan("b", "e"))
        self.assertEqual(result, {"b": "B", "c": "C", "d": "D", "e": "E"})

    def test_scan_excludes_deleted(self):
        db = LSMTree(memtable_limit=100)
        db.put("a", "1")
        db.put("b", "2")
        db.delete("a")
        result = dict(db.scan("a", "z"))
        self.assertEqual(result, {"b": "2"})


# ---------------------------------------------------------------------------
# Q-COL
# ---------------------------------------------------------------------------

class TestColumnStore(unittest.TestCase):
    def setUp(self):
        self.store = ColumnStore()
        self.store.add_column("city", ["Paris", "London", "Paris", "Berlin", "London"])
        self.store.add_column("sales", [100, 200, 150, 300, 250])
        self.store.add_column("qty", [1, 2, 1, 3, 2])

    def test_filter_eq(self):
        mask = self.store.filter("city", "==", "Paris")
        self.assertEqual(mask.count(), 2)
        self.assertEqual(mask.indices(), [0, 2])

    def test_filter_numeric(self):
        mask = self.store.filter("sales", ">", 150)
        self.assertEqual(mask.indices(), [1, 3, 4])

    def test_filter_and_or(self):
        mask = self.store.filter_and(("city", "==", "London"), ("sales", ">", 200))
        self.assertEqual(mask.indices(), [4])
        mask = self.store.filter_or(("city", "==", "Berlin"), ("sales", "==", 100))
        self.assertEqual(mask.indices(), [0, 3])

    def test_aggregate(self):
        self.assertEqual(self.store.aggregate("sales", "sum"), 1000)
        self.assertEqual(self.store.aggregate("sales", "min"), 100)
        self.assertEqual(self.store.aggregate("sales", "max"), 300)
        self.assertEqual(self.store.aggregate("sales", "avg"), 200)
        mask = self.store.filter("city", "==", "Paris")
        self.assertEqual(self.store.aggregate("sales", "sum", mask), 250)

    def test_groupby(self):
        result = self.store.groupby("city", {"sales": "sum", "qty": "avg"})
        by_city = {r["city"]: r for r in result}
        self.assertEqual(by_city["Paris"]["sales"], 250)
        self.assertEqual(by_city["London"]["sales"], 450)
        self.assertEqual(by_city["Berlin"]["sales"], 300)
        self.assertEqual(by_city["Paris"]["qty"], 1.0)

    def test_select_with_mask(self):
        mask = self.store.filter("city", "==", "Berlin")
        rows = self.store.select(["city", "sales"], mask)
        self.assertEqual(rows, [{"city": "Berlin", "sales": 300}])

    def test_mismatched_column_length_raises(self):
        with self.assertRaises(ValueError):
            self.store.add_column("bad", [1, 2])

    def test_dict_column_compression(self):
        col = DictColumn(["a", "a", "b", "a", "c", "b"] * 100)
        self.assertEqual(col.cardinality(), 3)
        self.assertEqual(len(col), 600)
        mask = col.filter_eq("b")
        self.assertEqual(mask.count(), 200)
        self.assertEqual(col.to_list()[:3], ["a", "a", "b"])

    def test_rle_column(self):
        col = RLEColumn([1, 1, 1, 2, 2, 3])
        self.assertEqual(len(col._runs), 3)
        self.assertEqual(col.sum(), 10)
        self.assertEqual(col.to_list(), [1, 1, 1, 2, 2, 3])

    def test_mask_operators(self):
        m1 = Mask([True, False, True])
        m2 = Mask([True, True, False])
        self.assertEqual((m1 & m2).bits, [True, False, False])
        self.assertEqual((m1 | m2).bits, [True, True, True])
        self.assertEqual((~m1).bits, [False, True, False])


# ---------------------------------------------------------------------------
# Q-STREAM
# ---------------------------------------------------------------------------

class TestStream(unittest.TestCase):
    def test_tumbling_window_sum(self):
        results = []
        proc = StreamProcessor(window_type="tumbling", window_size_ms=1000, agg="sum")
        proc.on_window(results.append)

        # Two events in window [0,1000), one in [1000,2000)
        proc.emit(Event(timestamp=100, key="s1", value=10))
        proc.emit(Event(timestamp=900, key="s1", value=20))
        proc.emit(Event(timestamp=1500, key="s1", value=5))
        proc.flush()

        by_start = {r.start: r for r in results}
        self.assertEqual(by_start[0].value, 30)
        self.assertEqual(by_start[1000].value, 5)

    def test_windows_fire_when_watermark_passes(self):
        results = []
        proc = StreamProcessor(window_type="tumbling", window_size_ms=1000)
        proc.on_window(results.append)

        proc.emit(Event(timestamp=100, key="s1", value=1))
        self.assertEqual(len(results), 0)        # window still open
        proc.emit(Event(timestamp=2500, key="s1", value=1))
        self.assertEqual(len(results), 1)        # [0,1000) closed by watermark
        self.assertEqual(results[0].start, 0)

    def test_keys_are_isolated(self):
        results = []
        proc = StreamProcessor(window_type="tumbling", window_size_ms=1000, agg="sum")
        proc.on_window(results.append)
        proc.emit(Event(timestamp=100, key="a", value=1))
        proc.emit(Event(timestamp=200, key="b", value=100))
        proc.flush()
        by_key = {r.key: r.value for r in results}
        self.assertEqual(by_key, {"a": 1, "b": 100})

    def test_late_events_dropped(self):
        proc = StreamProcessor(window_type="tumbling", window_size_ms=1000,
                               allowed_lateness_ms=0)
        proc.emit(Event(timestamp=5000, key="s1", value=1))
        proc.emit(Event(timestamp=100, key="s1", value=1))   # way too late
        self.assertEqual(proc.stats()["dropped_late_events"], 1)

    def test_allowed_lateness_accepts_recent_events(self):
        results = []
        proc = StreamProcessor(window_type="tumbling", window_size_ms=1000,
                               allowed_lateness_ms=2000, agg="count")
        proc.on_window(results.append)
        proc.emit(Event(timestamp=2500, key="s1", value=1))
        proc.emit(Event(timestamp=1500, key="s1", value=1))   # late but allowed
        proc.flush()
        total = sum(r.value for r in results)
        self.assertEqual(total, 2)

    def test_sliding_windows_overlap(self):
        results = []
        proc = StreamProcessor(window_type="sliding", window_size_ms=2000,
                               slide_ms=1000, agg="count")
        proc.on_window(results.append)
        proc.emit(Event(timestamp=1500, key="s1", value=1))
        proc.flush()
        # Event at t=1500 belongs to windows [0,2000) and [1000,3000)
        starts = sorted(r.start for r in results)
        self.assertEqual(starts, [0, 1000])

    def test_session_windows(self):
        results = []
        proc = SessionWindowProcessor(gap_ms=1000)
        proc.on_window(results.append)
        # Burst 1: t=0,200,400 — Burst 2 (gap>1000): t=3000
        for t in (0, 200, 400, 3000):
            proc.emit(Event(timestamp=t, key="u1", value=1))
        proc.flush()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].count, 3)
        self.assertEqual(results[1].count, 1)


# ---------------------------------------------------------------------------
# Q-MEM
# ---------------------------------------------------------------------------

class TestArena(unittest.TestCase):
    def test_size_classes(self):
        classes = build_size_classes(4096)
        self.assertEqual(size_class_for(1, classes), 8)
        self.assertEqual(size_class_for(8, classes), 8)
        self.assertEqual(size_class_for(9, classes), 16)
        self.assertEqual(size_class_for(100, classes), 112)

    def test_alloc_write_read(self):
        arena = ArenaAllocator(slab_size=4096)
        a = arena.alloc(100)
        arena.write(a, b"hello world")
        self.assertEqual(arena.read(a, 11), b"hello world")

    def test_allocations_are_isolated(self):
        arena = ArenaAllocator(slab_size=4096)
        a = arena.alloc(50)
        b = arena.alloc(50)
        arena.write(a, b"AAAA")
        arena.write(b, b"BBBB")
        self.assertEqual(arena.read(a, 4), b"AAAA")
        self.assertEqual(arena.read(b, 4), b"BBBB")

    def test_free_and_reuse(self):
        arena = ArenaAllocator(slab_size=4096)
        keep = arena.alloc(64)  # keeps the slab alive after `a` is freed
        a = arena.alloc(64)
        slab_id, block_idx = a.slab_id, a.block_idx
        arena.free(a)
        b = arena.alloc(64)
        # LIFO free list should reuse the same block
        self.assertEqual((b.slab_id, b.block_idx), (slab_id, block_idx))
        arena.free(keep)

    def test_stats_track_fragmentation(self):
        arena = ArenaAllocator(slab_size=4096)
        a = arena.alloc(100)  # rounds to 112 → 12 bytes internal frag
        s = arena.stats()
        self.assertEqual(s["bytes_requested"], 100)
        self.assertEqual(s["bytes_allocated"], 112)
        self.assertEqual(s["internal_fragmentation_bytes"], 12)
        self.assertEqual(a.internal_fragmentation, 12)

    def test_reset_frees_everything(self):
        arena = ArenaAllocator(slab_size=4096)
        for _ in range(100):
            arena.alloc(64)
        arena.reset()
        s = arena.stats()
        self.assertEqual(s["live_allocations"], 0)
        self.assertEqual(s["num_slabs"], 0)

    def test_large_object_gets_own_slab(self):
        arena = ArenaAllocator(slab_size=4096)
        big = arena.alloc(100_000)
        self.assertEqual(big.block_size, 100_000)
        arena.write(big, b"x" * 100_000)
        self.assertEqual(len(arena.read(big, 100_000)), 100_000)

    def test_empty_slab_released_on_free(self):
        arena = ArenaAllocator(slab_size=4096)
        a = arena.alloc(64)
        self.assertEqual(arena.stats()["num_slabs"], 1)
        arena.free(a)
        self.assertEqual(arena.stats()["num_slabs"], 0)

    def test_write_overflow_raises(self):
        arena = ArenaAllocator(slab_size=4096)
        a = arena.alloc(8)
        with self.assertRaises(ValueError):
            arena.write(a, b"x" * 100)

    def test_many_allocations_spill_to_new_slabs(self):
        arena = ArenaAllocator(slab_size=256)  # tiny slabs: 4 blocks of 64
        allocs = [arena.alloc(64) for _ in range(10)]
        self.assertGreaterEqual(arena.stats()["num_slabs"], 3)
        # All blocks remain readable/writable
        for i, a in enumerate(allocs):
            arena.write(a, bytes([i] * 8))
        for i, a in enumerate(allocs):
            self.assertEqual(arena.read(a, 8), bytes([i] * 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
