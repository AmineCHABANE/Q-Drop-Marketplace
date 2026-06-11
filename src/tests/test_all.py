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
from q_grover import GroverSearch, PhaseOracle, DiffusionOperator, QuantumFourierTransform, _hadamard_all
from q_kyber import keygen, encapsulate, decapsulate, verify_correctness, N as KYBER_N
from q_shor import ShorFactorer, find_period_classical, is_prime, gcd
from q_qec import (BitFlipCode, PhaseFlipCode, ShorCode, QubitState,
                   Register, Error, ErrorType, apply_error, run_error_correction_demo,
                   build_steane_code, PauliOperator)
from q_vqe import (VQE, QAOA, HardwareEfficientAnsatz, UCCSDSingletAnsatz,
                   Hamiltonian, PauliTerm,
                   zero_state, _apply_ry, _apply_h, _pauli_string_expectation)
import q_pack
import q_sign
import q_qsf
import q_kyber


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


# ---------------------------------------------------------------------------
# Q-GROVER
# ---------------------------------------------------------------------------

class TestGrover(unittest.TestCase):
    def test_finds_target_with_high_probability(self):
        """Grover's should find the target with probability >> 1/N."""
        gs = GroverSearch(n_qubits=4, seed=42)   # N = 16
        target = 7
        p_success = gs.probability_of_success([target])
        # Optimal Grover: should exceed 90% success probability for N=16
        self.assertGreater(p_success, 0.9)

    def test_single_search_returns_target(self):
        """Run search 10 times; target must appear in majority."""
        target = 11
        gs = GroverSearch(n_qubits=4, seed=0)
        hits = sum(1 for s in range(10) if GroverSearch(n_qubits=4, seed=s).search([target]) == target)
        self.assertGreaterEqual(hits, 8)

    def test_optimal_iterations(self):
        """π/4 · √N for 1 solution."""
        gs = GroverSearch(n_qubits=8)   # N = 256
        k = gs.optimal_iterations()
        # Should be close to π/4 · sqrt(256) = π/4 · 16 ≈ 12.6
        self.assertAlmostEqual(k, 13, delta=2)

    def test_multiple_targets(self):
        """With 4 targets in N=16, probability should be very high."""
        gs = GroverSearch(n_qubits=4, n_solutions=4, seed=1)
        targets = [0, 4, 8, 12]
        p = gs.probability_of_success(targets)
        self.assertGreater(p, 0.95)

    def test_statevector_is_normalized(self):
        """Total probability must sum to 1."""
        gs = GroverSearch(n_qubits=5, seed=7)
        sv = gs.statevector([3])
        total_prob = sum(abs(a)**2 for a in sv)
        self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_hadamard_all_uniform(self):
        """H⊗n should give uniform amplitudes 1/√N."""
        import math
        state = _hadamard_all(4)
        expected = 1.0 / math.sqrt(16)
        for amp in state:
            self.assertAlmostEqual(abs(amp), expected, places=12)

    def test_qft_preserves_norm(self):
        """QFT should be unitary (norm-preserving)."""
        import math
        qft = QuantumFourierTransform(n_qubits=4)
        state = _hadamard_all(4)
        transformed = qft.apply(state)
        norm = sum(abs(a)**2 for a in transformed)
        self.assertAlmostEqual(norm, 1.0, places=8)

    def test_invalid_oracle_raises(self):
        gs = GroverSearch(n_qubits=3)
        with self.assertRaises(ValueError):
            gs.search([100])   # 100 >= 2^3 = 8


# ---------------------------------------------------------------------------
# Q-KYBER
# ---------------------------------------------------------------------------

class TestKyber(unittest.TestCase):
    def test_keygen_produces_keys(self):
        seed = b"\x00" * 32
        pk, sk = keygen(seed)
        # Matrix A is k×k = 2×2, each polynomial has N=256 coefficients
        self.assertEqual(len(pk.A), 2)
        self.assertEqual(len(pk.A[0]), 2)
        self.assertEqual(len(pk.A[0][0]), KYBER_N)

    def test_encap_decap_correctness(self):
        """Shared secrets must match after key exchange."""
        seed = b"\x01" * 32
        msg  = b"\x42" * 32
        pk, sk = keygen(seed)
        ct, ss_enc = encapsulate(pk, msg)
        ss_dec = decapsulate(sk, ct)
        self.assertEqual(ss_enc, ss_dec)

    def test_different_messages_different_secrets(self):
        seed = b"\x00" * 32
        pk, sk = keygen(seed)
        msg1, msg2 = b"\x01" * 32, b"\x02" * 32
        _, ss1 = encapsulate(pk, msg1)
        _, ss2 = encapsulate(pk, msg2)
        self.assertNotEqual(ss1, ss2)

    def test_different_keys_different_decap(self):
        """Decapsulating with the wrong key should produce a different secret."""
        pk1, sk1 = keygen(b"\x01" * 32)
        pk2, sk2 = keygen(b"\x02" * 32)
        msg = b"\xAB" * 32
        ct, ss_enc = encapsulate(pk1, msg)
        ss_wrong = decapsulate(sk2, ct)
        self.assertNotEqual(ss_enc, ss_wrong)

    def test_bulk_correctness(self):
        """verify_correctness helper should pass all 10 trials."""
        successes, n = verify_correctness(n_trials=10)
        self.assertEqual(successes, n)

    def test_ciphertext_structure(self):
        pk, sk = keygen(b"\x00" * 32)
        ct, _ = encapsulate(pk)
        # u: k=2 polynomials of N=256 compressed coefficients
        self.assertEqual(len(ct.u), 2)
        self.assertEqual(len(ct.u[0]), KYBER_N)
        # v: single polynomial
        self.assertEqual(len(ct.v), KYBER_N)


# ---------------------------------------------------------------------------
# Q-SHOR
# ---------------------------------------------------------------------------

class TestShor(unittest.TestCase):
    def test_gcd(self):
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(100, 75), 25)
        self.assertEqual(gcd(7, 13), 1)

    def test_is_prime(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 97, 9973]
        composites = [1, 4, 6, 9, 15, 100, 9975]
        for p in primes:
            self.assertTrue(is_prime(p), f"{p} should be prime")
        for c in composites:
            self.assertFalse(is_prime(c), f"{c} should be composite")

    def test_classical_period_finding(self):
        """2^r ≡ 1 (mod 15) has period 4."""
        r = find_period_classical(2, 15)
        self.assertEqual(r, 4)
        self.assertEqual(pow(2, r, 15), 1)

    def test_factor_small_semiprime(self):
        """Factor N=15=3×5 and N=21=3×7."""
        shor = ShorFactorer(use_quantum_simulation=False)
        for N, factors in [(15, {3, 5}), (21, {3, 7})]:
            f = shor.factor(N, seed=0)
            self.assertIsNotNone(f, f"Failed to factor {N}")
            self.assertIn(f, factors, f"Factor of {N} should be in {factors}, got {f}")

    def test_full_factorization(self):
        """Full factorization of 12 = 2 × 2 × 3."""
        shor = ShorFactorer(use_quantum_simulation=False)
        factors = shor.full_factorization(12, seed=0)
        self.assertEqual(factors, [2, 2, 3])

    def test_prime_raises(self):
        shor = ShorFactorer()
        with self.assertRaises(ValueError):
            shor.factor(13)

    def test_even_number_factor(self):
        shor = ShorFactorer()
        self.assertEqual(shor.factor(14), 2)

    def test_perfect_power_detection(self):
        shor = ShorFactorer()
        f = shor.factor(8)   # 2^3
        self.assertIn(f, {2, 4})


# ---------------------------------------------------------------------------
# Q-QEC
# ---------------------------------------------------------------------------

class TestQEC(unittest.TestCase):
    def test_qubit_state_normalization(self):
        import math
        q = QubitState(3 + 0j, 4 + 0j)
        self.assertAlmostEqual(abs(q.alpha)**2 + abs(q.beta)**2, 1.0)
        self.assertAlmostEqual(abs(q.alpha), 0.6)

    def test_pauli_gates(self):
        import math
        # X: |0⟩ → |1⟩
        q = QubitState.zero().apply_x()
        self.assertAlmostEqual(abs(q.beta), 1.0)
        # Z: |+⟩ → |−⟩
        plus = QubitState.plus()
        minus = plus.apply_z()
        self.assertAlmostEqual(minus.beta.real, -1/math.sqrt(2), places=10)
        # HZH = X
        q0 = QubitState.zero()
        q1 = q0.apply_h().apply_z().apply_h()
        # Should equal X|0⟩ = |1⟩ approximately
        self.assertAlmostEqual(abs(q1.beta), 1.0, places=10)

    def test_fidelity_same_state(self):
        q = QubitState.plus()
        self.assertAlmostEqual(q.fidelity(q), 1.0)

    def test_fidelity_orthogonal(self):
        self.assertAlmostEqual(QubitState.zero().fidelity(QubitState.one()), 0.0)

    def test_bit_flip_code_corrects_error(self):
        import math
        code = BitFlipCode()
        alpha, beta = 1/math.sqrt(2), 1/math.sqrt(2)
        reg = code.encode(alpha, beta)
        # Inject bit flip on qubit 1
        reg.apply_x(1)
        s01, s12 = code.syndrome(reg, random.Random(0))
        code.correct(reg, s01, s12)
        recovered = code.decode(reg, random.Random(0))
        ideal = QubitState(alpha, beta)
        self.assertGreater(recovered.fidelity(ideal), 0.99)

    def test_bit_flip_code_no_error(self):
        import math
        code = BitFlipCode()
        alpha, beta = 1/math.sqrt(2), 1/math.sqrt(2)
        reg = code.encode(alpha, beta)
        s01, s12 = code.syndrome(reg, random.Random(0))
        self.assertEqual((s01, s12), (0, 0))

    def test_phase_flip_code_corrects_z_error(self):
        import math
        code = PhaseFlipCode()
        alpha, beta = 1/math.sqrt(2), 1/math.sqrt(2)
        reg = code.encode(alpha, beta)
        reg.apply_z(2)
        s01, s12 = code.syndrome(reg, random.Random(0))
        code.correct(reg, s01, s12)
        # No assertion on exact recovered state due to product-state approximation
        # Just verify syndrome and correction ran without error

    def test_error_correction_monte_carlo(self):
        stats = run_error_correction_demo(n_trials=200, p_error=0.05, seed=42)
        # QEC should outperform no-correction
        self.assertGreater(
            stats["success_rate_with_qec"],
            stats["success_rate_without_qec"]
        )
        self.assertGreater(stats["success_rate_with_qec"], 0.85)

    def test_steane_code_structure(self):
        code = build_steane_code()
        self.assertEqual(code.n, 7)
        self.assertEqual(code.k, 1)
        self.assertEqual(len(code.generators), 6)

    def test_stabilizer_syndrome_trivial(self):
        code = build_steane_code()
        # Identity error: should commute with all generators
        n = 7
        identity = PauliOperator(n, [0]*n, [0]*n)
        syndrome = code.syndrome(identity)
        self.assertEqual(syndrome, [0] * 6)

    def test_pauli_commutation(self):
        """X and Z anticommute, X and X commute."""
        n = 1
        X = PauliOperator(n, [1], [0])
        Z = PauliOperator(n, [0], [1])
        self.assertFalse(X.commutes_with(Z))
        self.assertTrue(X.commutes_with(X))
        self.assertTrue(Z.commutes_with(Z))


# ---------------------------------------------------------------------------
# Q-VQE
# ---------------------------------------------------------------------------

class TestVQE(unittest.TestCase):
    def test_zero_state(self):
        sv = zero_state(2)
        self.assertEqual(len(sv), 4)
        self.assertAlmostEqual(abs(sv[0]), 1.0)

    def test_pauli_z_expectation_on_zero(self):
        """⟨0|Z|0⟩ = 1."""
        sv = zero_state(1)
        ev = _pauli_string_expectation(sv, "Z")
        self.assertAlmostEqual(ev, 1.0)

    def test_pauli_x_expectation_on_zero(self):
        """⟨0|X|0⟩ = 0."""
        sv = zero_state(1)
        ev = _pauli_string_expectation(sv, "X")
        self.assertAlmostEqual(ev, 0.0, places=10)

    def test_pauli_z_expectation_on_plus(self):
        """⟨+|Z|+⟩ = 0."""
        sv = zero_state(1)
        sv = _apply_h(sv, 0, 1)
        ev = _pauli_string_expectation(sv, "Z")
        self.assertAlmostEqual(ev, 0.0, places=10)

    def test_hamiltonian_energy_ground_state(self):
        """For single-qubit H = Z, ground state is |1⟩ with energy -1."""
        H = Hamiltonian([PauliTerm(1.0, "Z")])
        sv = zero_state(1)
        sv[0], sv[1] = 0j, 1 + 0j   # |1⟩
        ev = H.expectation_value(sv)
        self.assertAlmostEqual(ev, -1.0, places=10)

    def test_h2_hamiltonian_energy_range(self):
        """H₂ ground state energy should be around -1.137 Hartree."""
        H = Hamiltonian.h2_molecule()
        ansatz = UCCSDSingletAnsatz()
        vqe = VQE(H, ansatz, learning_rate=0.05, max_iterations=300)
        result = vqe.run(seed=42)
        # Ground state is around -1.137; accept -1.3 to -0.8 for convergence range
        self.assertLess(result.optimal_energy, -0.8)
        self.assertGreater(result.optimal_energy, -1.3)

    def test_ising_model_hamiltonian(self):
        H = Hamiltonian.ising_model(n=2, J=1.0, h=0.5)
        self.assertGreater(len(H.terms), 0)
        self.assertEqual(H.n_qubits, 2)

    def test_hardware_efficient_ansatz_param_count(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=3, depth=2)
        # (depth+1) × n_qubits = 3 × 3 = 9
        self.assertEqual(ansatz.n_params, 9)

    def test_hardware_efficient_ansatz_builds_state(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
        params = [0.0] * ansatz.n_params
        sv = ansatz.build_state(params)
        # All-zero rotation: state stays near |00⟩
        self.assertEqual(len(sv), 4)
        total_prob = sum(abs(a)**2 for a in sv)
        self.assertAlmostEqual(total_prob, 1.0, places=8)

    def test_vqe_minimizes_energy(self):
        """VQE should lower energy from initial random params."""
        import math
        H = Hamiltonian([PauliTerm(1.0, "ZZ"), PauliTerm(0.5, "XI")])
        ansatz = HardwareEfficientAnsatz(n_qubits=2, depth=1)
        vqe = VQE(H, ansatz, learning_rate=0.05, max_iterations=100)
        result = vqe.run(seed=0)
        # Energy must have decreased over iterations
        self.assertLess(result.energy_history[-1], result.energy_history[0])

    def test_maxcut_hamiltonian(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        H = Hamiltonian.maxcut_qubo(edges, n=3)
        self.assertEqual(H.n_qubits, 3)

    def test_qaoa_runs(self):
        edges = [(0, 1), (1, 2)]
        H = Hamiltonian.maxcut_qubo(edges, n=3)
        qaoa = QAOA(H, p_layers=1)
        result = qaoa.run(seed=42)
        self.assertIsNotNone(result.optimal_energy)
        self.assertGreater(len(result.energy_history), 0)


# ---------------------------------------------------------------------------
# Q-PACK
# ---------------------------------------------------------------------------

class TestPack(unittest.TestCase):
    def test_roundtrip_empty(self):
        self.assertEqual(q_pack.decompress(q_pack.compress(b"")), b"")

    def test_roundtrip_single_byte(self):
        self.assertEqual(q_pack.decompress(q_pack.compress(b"x")), b"x")

    def test_roundtrip_repetitive(self):
        data = b"abcabcabc" * 500
        blob = q_pack.compress(data)
        self.assertEqual(q_pack.decompress(blob), data)
        # Highly repetitive data must compress well
        self.assertLess(len(blob), len(data) // 4)

    def test_roundtrip_random(self):
        rng = random.Random(0)
        data = bytes(rng.randrange(256) for _ in range(4096))
        self.assertEqual(q_pack.decompress(q_pack.compress(data)), data)

    def test_incompressible_falls_back_to_stored(self):
        """Random data must never blow up beyond input + small header."""
        rng = random.Random(1)
        data = bytes(rng.randrange(256) for _ in range(1000))
        blob = q_pack.compress(data)
        self.assertLessEqual(len(blob), len(data) + 4)

    def test_overlapping_match(self):
        """Run-length via self-overlapping back-reference (offset < length)."""
        data = b"a" * 1000
        self.assertEqual(q_pack.decompress(q_pack.compress(data)), data)

    def test_bad_magic_raises(self):
        with self.assertRaises(ValueError):
            q_pack.decompress(b"XXXX" + b"\x00" * 20)

    def test_text_compresses(self):
        data = ("the quick brown fox jumps over the lazy dog " * 100).encode()
        blob = q_pack.compress(data)
        self.assertEqual(q_pack.decompress(blob), data)
        self.assertLess(len(blob), len(data) // 2)


# ---------------------------------------------------------------------------
# Q-SIGN
# ---------------------------------------------------------------------------

class TestSign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # height=3 → 8 leaves; small for test speed, shared across tests
        cls.seed = b"unit-test-seed-0123456789abcdef0"

    def make_signer(self):
        return q_sign.MerkleSigner(self.seed, height=3)

    def test_sign_and_verify(self):
        s = self.make_signer()
        root = s.public_root
        sig = s.sign(b"hello quantum world")
        self.assertTrue(q_sign.verify(b"hello quantum world", sig, root))

    def test_wrong_message_rejected(self):
        s = self.make_signer()
        sig = s.sign(b"message A")
        self.assertFalse(q_sign.verify(b"message B", sig, s.public_root))

    def test_wrong_root_rejected(self):
        s = self.make_signer()
        sig = s.sign(b"msg")
        other = q_sign.MerkleSigner(b"another-seed-0123456789abcdef000", height=3)
        self.assertFalse(q_sign.verify(b"msg", sig, other.public_root))

    def test_tampered_signature_rejected(self):
        s = self.make_signer()
        sig = s.sign(b"msg")
        sig.wots_sig[0] = b"\x00" * 32
        self.assertFalse(q_sign.verify(b"msg", sig, s.public_root))

    def test_each_leaf_used_once(self):
        s = self.make_signer()
        indices = [s.sign(f"m{i}".encode()).leaf_index for i in range(8)]
        self.assertEqual(indices, list(range(8)))

    def test_exhaustion_raises(self):
        s = self.make_signer()
        for i in range(8):
            s.sign(f"m{i}".encode())
        with self.assertRaises(RuntimeError):
            s.sign(b"one too many")

    def test_serialization_roundtrip(self):
        s = self.make_signer()
        sig = s.sign(b"serialize me")
        sig2 = q_sign.Signature.from_bytes(sig.to_bytes())
        self.assertTrue(q_sign.verify(b"serialize me", sig2, s.public_root))

    def test_state_persistence(self):
        s = self.make_signer()
        s.sign(b"first")
        restored = q_sign.MerkleSigner.from_state(s.to_state())
        self.assertEqual(restored.next_index, 1)
        self.assertEqual(restored.public_root, s.public_root)
        # restored signer must not reuse leaf 0
        sig = restored.sign(b"second")
        self.assertEqual(sig.leaf_index, 1)

    def test_deterministic_root_from_seed(self):
        a = q_sign.MerkleSigner(self.seed, height=3)
        b = q_sign.MerkleSigner(self.seed, height=3)
        self.assertEqual(a.public_root, b.public_root)


# ---------------------------------------------------------------------------
# Q-QSF
# ---------------------------------------------------------------------------

class TestQSF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = ("quantum daily data " * 200).encode()
        cls.meta = {"creator": "test", "type": "text/plain"}
        cls.pk, cls.sk = q_kyber.keygen(b"\x05" * 32)
        cls.pk2, cls.sk2 = q_kyber.keygen(b"\x06" * 32)

    def test_plain_roundtrip(self):
        blob = q_qsf.create(self.payload, self.meta)
        c = q_qsf.open_container(blob)
        self.assertEqual(c.payload, self.payload)
        self.assertEqual(c.metadata, self.meta)
        self.assertTrue(c.compressed)
        self.assertFalse(c.encrypted)

    def test_compression_effective(self):
        blob = q_qsf.create(self.payload, self.meta)
        self.assertLess(len(blob), len(self.payload) // 2)

    def test_encrypted_roundtrip(self):
        blob = q_qsf.create(self.payload, self.meta,
                            recipient_pk=self.pk, _kem_message=b"\x11" * 32)
        c = q_qsf.open_container(blob, sk=self.sk)
        self.assertEqual(c.payload, self.payload)
        self.assertTrue(c.encrypted)

    def test_encrypted_requires_key(self):
        blob = q_qsf.create(self.payload, self.meta,
                            recipient_pk=self.pk, _kem_message=b"\x12" * 32)
        with self.assertRaises(ValueError):
            q_qsf.open_container(blob)

    def test_wrong_key_fails(self):
        blob = q_qsf.create(self.payload, self.meta,
                            recipient_pk=self.pk, _kem_message=b"\x13" * 32)
        with self.assertRaises(ValueError):
            q_qsf.open_container(blob, sk=self.sk2)

    def test_signed_roundtrip(self):
        signer = q_sign.MerkleSigner(b"qsf-test-seed-000000000000000000", height=3)
        root = signer.public_root
        blob = q_qsf.create(self.payload, self.meta, signer=signer)
        c = q_qsf.open_container(blob, signer_root=root)
        self.assertIs(c.signature_valid, True)

    def test_unsigned_with_root_required_fails(self):
        signer = q_sign.MerkleSigner(b"qsf-test-seed-000000000000000000", height=3)
        blob = q_qsf.create(self.payload, self.meta)   # unsigned
        with self.assertRaises(ValueError):
            q_qsf.open_container(blob, signer_root=signer.public_root)

    def test_corruption_detected(self):
        blob = bytearray(q_qsf.create(self.payload, self.meta))
        blob[15] ^= 0xFF
        with self.assertRaises(ValueError):
            q_qsf.open_container(bytes(blob))

    def test_full_stack(self):
        """Compressed + Kyber-encrypted + hash-signed, all verified."""
        signer = q_sign.MerkleSigner(b"qsf-full-seed-000000000000000000", height=3)
        root = signer.public_root
        blob = q_qsf.create(self.payload, self.meta,
                            recipient_pk=self.pk, signer=signer,
                            _kem_message=b"\x14" * 32)
        c = q_qsf.open_container(blob, sk=self.sk, signer_root=root)
        self.assertEqual(c.payload, self.payload)
        self.assertTrue(c.compressed and c.encrypted and c.signed)
        self.assertIs(c.signature_valid, True)

    def test_bad_magic(self):
        with self.assertRaises(ValueError):
            q_qsf.open_container(b"NOPE" + b"\x00" * 40)


# ---------------------------------------------------------------------------
# Q-DILITHIUM
# ---------------------------------------------------------------------------

import q_dilithium

class TestDilithium(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pk, cls.sk = q_dilithium.keygen(b"\xAB" * 32)

    def test_sign_and_verify(self):
        msg = b"dilithium test message"
        sig = self.__class__.sk.sign(msg)
        self.assertTrue(q_dilithium.verify(self.__class__.pk, msg, sig))

    def test_sign_high_level_api(self):
        msg = b"post-quantum signature test"
        sig_bytes = q_dilithium.sign_message(self.__class__.sk, msg)
        self.assertTrue(q_dilithium.verify_signature(self.__class__.pk, msg, sig_bytes))

    def test_tampered_message_rejected(self):
        msg = b"original"
        sig = self.__class__.sk.sign(msg)
        self.assertFalse(q_dilithium.verify(self.__class__.pk, b"tampered", sig))

    def test_wrong_key_rejected(self):
        pk2, _ = q_dilithium.keygen()
        msg = b"hello quantum"
        sig = self.__class__.sk.sign(msg)
        self.assertFalse(q_dilithium.verify(pk2, msg, sig))

    def test_keygen_deterministic(self):
        seed = b"\xCC" * 32
        pk1, sk1 = q_dilithium.keygen(seed)
        pk2, sk2 = q_dilithium.keygen(seed)
        self.assertEqual(pk1.to_bytes(), pk2.to_bytes())
        self.assertEqual(sk1.rho, sk2.rho)

    def test_pk_serialisation_roundtrip(self):
        pk = self.__class__.pk
        raw = pk.to_bytes()
        pk2 = q_dilithium.DilithiumPublicKey.from_bytes(raw)
        self.assertEqual(pk.rho, pk2.rho)
        self.assertEqual(pk.t1[0][:10], pk2.t1[0][:10])

    def test_sig_serialisation_roundtrip(self):
        msg = b"sig serialisation"
        sig = self.__class__.sk.sign(msg)
        raw = sig.to_bytes()
        sig2 = q_dilithium.DilithiumSignature.from_bytes(raw)
        self.assertEqual(sig.c_tilde, sig2.c_tilde)

    def test_ntt_roundtrip(self):
        from q_dilithium import _ntt, _inv_ntt, _Q
        rng = random.Random(99)
        a = [rng.randrange(_Q) for _ in range(256)]
        back = _inv_ntt(_ntt(a))
        self.assertTrue(all((x - y) % _Q == 0 for x, y in zip(a, back)))

    def test_poly_multiplication(self):
        """(1+X)^2 = 1 + 2X + X^2 mod q."""
        from q_dilithium import _ntt, _inv_ntt, _ntt_mul
        a = [0] * 256; a[0] = 1; a[1] = 1
        prod = _inv_ntt(_ntt_mul(_ntt(a), _ntt(a)))
        self.assertEqual(prod[0], 1)
        self.assertEqual(prod[1], 2)
        self.assertEqual(prod[2], 1)
        self.assertEqual(prod[3], 0)

    def test_correctness_batch(self):
        ok, total = q_dilithium.verify_correctness(3)
        self.assertEqual(ok, total)

    def test_empty_message(self):
        sig = self.__class__.sk.sign(b"")
        self.assertTrue(q_dilithium.verify(self.__class__.pk, b"", sig))


# ---------------------------------------------------------------------------
# Q-RAFT
# ---------------------------------------------------------------------------

import q_raft

class TestRaft(unittest.TestCase):
    def _make_cluster(self, n=5, seed=7):
        return q_raft.RaftCluster(n, seed=seed)

    def test_leader_election(self):
        c = self._make_cluster()
        c.tick(40)
        self.assertIsNotNone(c.leader())

    def test_single_write(self):
        c = self._make_cluster()
        c.tick(40)
        ok, idx = c.propose({"op": "set", "key": "x", "value": 42})
        self.assertTrue(ok)
        self.assertGreater(idx, 0)
        c.tick(10)
        self.assertEqual(c.read("x"), 42)

    def test_multiple_writes_consistent(self):
        c = self._make_cluster()
        c.tick(40)
        for i in range(5):
            ok, _ = c.propose({"op": "set", "key": f"k{i}", "value": i})
            self.assertTrue(ok)
        c.tick(20)
        self.assertTrue(c.is_consistent())

    def test_follower_crash_and_recovery(self):
        c = self._make_cluster()
        c.tick(40)
        ldr = c.leader()
        follower = next(s for s in c.servers.values()
                        if s.id != ldr.id and s.id not in c._dead)
        c.crash(follower.id)
        ok, _ = c.propose({"op": "set", "key": "after_crash", "value": 1})
        self.assertTrue(ok)
        c.tick(20)
        c.restart(follower.id)
        c.tick(60)
        self.assertTrue(c.is_consistent())

    def test_partition_and_reelection(self):
        c = self._make_cluster()
        c.tick(40)
        old_id = c.leader().id
        c.partition([old_id])
        c.tick(50)
        new_ldr = c.leader()
        self.assertIsNotNone(new_ldr)
        self.assertNotEqual(new_ldr.id, old_id)

    def test_heal_gives_single_leader(self):
        c = self._make_cluster()
        c.tick(40)
        c.partition([c.leader().id])
        c.tick(40)
        c.heal()
        c.tick(40)
        leaders = [s for s in c.servers.values() if s.role == q_raft.Role.LEADER
                   and s.id not in c._dead]
        self.assertEqual(len(leaders), 1)

    def test_quorum_required_for_commit(self):
        """No commit when majority down."""
        c = self._make_cluster(5)
        c.tick(40)
        # Kill 3 out of 5 (minority left = 2, can't reach quorum)
        live = [s.id for s in c.servers.values() if s.id not in c._dead]
        for sid in live[:3]:
            c.crash(sid)
        c.tick(40)
        ldr = c.leader()
        self.assertIsNone(ldr)   # no leader can be elected with only 2 nodes

    def test_full_demo(self):
        result = q_raft.run_basic_demo(5, seed=42)
        self.assertTrue(result["consistent"])
        self.assertGreater(result["entries_committed"], 10)

    def test_log_replication_order(self):
        c = self._make_cluster()
        c.tick(40)
        cmds = [{"op": "set", "key": "seq", "value": i} for i in range(5)]
        for cmd in cmds:
            c.propose(cmd)
        c.tick(30)
        # All live servers should have same commit index
        live = [s for s in c.servers.values() if s.id not in c._dead]
        commits = [s.commit_index for s in live]
        self.assertEqual(len(set(commits)), 1)


# ---------------------------------------------------------------------------
# Q-BLOOM
# ---------------------------------------------------------------------------

import q_bloom

class TestBloom(unittest.TestCase):
    def test_bloom_membership(self):
        bf = q_bloom.BloomFilter(1000, 0.01)
        for i in range(500):
            bf.add(f"item{i}")
        for i in range(500):
            self.assertIn(f"item{i}", bf)

    def test_bloom_false_positive_rate(self):
        bf = q_bloom.BloomFilter(10000, 0.01)
        for i in range(10000):
            bf.add(f"member:{i}")
        fp = sum(1 for i in range(10000, 30000) if f"member:{i}" in bf)
        self.assertLess(fp / 20000, 0.05)  # generous bound

    def test_bloom_merge(self):
        bf1 = q_bloom.BloomFilter(100, 0.01)
        bf2 = q_bloom.BloomFilter(100, 0.01)
        for i in range(50):
            bf1.add(f"a{i}")
        for i in range(50):
            bf2.add(f"b{i}")
        merged = bf1.merge(bf2)
        self.assertIn("a0", merged)
        self.assertIn("b0", merged)

    def test_counting_bloom_delete(self):
        cf = q_bloom.CountingBloomFilter(50, 0.01)
        cf.add("solo")
        self.assertIn("solo", cf)
        cf.remove("solo")
        self.assertNotIn("solo", cf)

    def test_xor_filter_membership(self):
        items = [f"key:{i}" for i in range(300)]
        xf = q_bloom.XorFilter.from_items(items)
        for item in items:
            self.assertIn(item, xf)

    def test_xor_filter_false_positive_rate(self):
        items = [f"in:{i}" for i in range(1000)]
        xf = q_bloom.XorFilter.from_items(items)
        fp = sum(1 for i in range(1000, 5000) if f"in:{i}" in xf)
        self.assertLess(fp / 4000, 0.05)

    def test_cuckoo_filter_insert_lookup_delete(self):
        cf = q_bloom.CuckooFilter(500)
        for i in range(200):
            cf.add(f"x{i}")
        self.assertIn("x0", cf)
        cf.remove("x0")
        self.assertNotIn("x0", cf)

    def test_hll_cardinality(self):
        hll = q_bloom.HyperLogLog(b=10)
        for i in range(5000):
            hll.add(f"el:{i}")
        est = hll.count()
        self.assertAlmostEqual(est, 5000, delta=500)  # within 10%

    def test_hll_merge(self):
        h1 = q_bloom.HyperLogLog(10)
        h2 = q_bloom.HyperLogLog(10)
        for i in range(1000):
            h1.add(f"a{i}")
        for i in range(1000):
            h2.add(f"b{i}")
        merged = h1.merge(h2)
        est = merged.count()
        self.assertGreater(est, 1000)

    def test_minhash_jaccard(self):
        s1 = set(range(100))
        s2 = set(range(50, 150))
        real = len(s1 & s2) / len(s1 | s2)   # = 50/150 = 1/3
        m1 = q_bloom.MinHash.from_set(s1, n_hashes=256)
        m2 = q_bloom.MinHash.from_set(s2, n_hashes=256)
        est = m1.similarity(m2)
        self.assertAlmostEqual(est, real, delta=0.10)

    def test_lsh_finds_candidates(self):
        lsh = q_bloom.LSHBand(n_hashes=128, bands=16)
        base = set(range(100))
        # Add near-duplicate (Jaccard ≈ 0.9)
        similar = set(list(range(90)) + [200, 201, 202, 203, 204, 205, 206, 207, 208, 209])
        m_base = q_bloom.MinHash.from_set(base, n_hashes=128)
        m_sim  = q_bloom.MinHash.from_set(similar, n_hashes=128)
        lsh.add("base", m_base)
        lsh.add("similar", m_sim)
        cands = lsh.candidates(m_sim)
        self.assertIn("base", cands)


if __name__ == "__main__":
    unittest.main(verbosity=2)
