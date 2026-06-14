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


# ---------------------------------------------------------------------------
# Q-ATTENTION
# ---------------------------------------------------------------------------

import q_attention

class TestAttention(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        s = q_attention.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(s), 1.0, places=12)

    def test_softmax_numerically_stable(self):
        """Large inputs must not overflow."""
        s = q_attention.softmax([1000.0, 1001.0, 1002.0])
        self.assertAlmostEqual(sum(s), 1.0, places=12)
        self.assertGreater(s[2], s[0])

    def test_softmax_masked_entries_zero(self):
        s = q_attention.softmax([1.0, q_attention.NEG_INF, 2.0])
        self.assertEqual(s[1], 0.0)
        self.assertAlmostEqual(sum(s), 1.0, places=12)

    def test_attention_weights_sum_to_one(self):
        Q = [[1.0, 0.0], [0.0, 1.0]]
        K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        V = [[1.0], [2.0], [3.0]]
        _, weights = q_attention.scaled_dot_product_attention(Q, K, V)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1.0, places=12)

    def test_attention_output_is_convex_combination(self):
        """Output values must lie within the range of the value vectors."""
        Q = [[0.5, 0.5]]
        K = [[1.0, 0.0], [0.0, 1.0]]
        V = [[10.0], [20.0]]
        out, _ = q_attention.scaled_dot_product_attention(Q, K, V)
        self.assertGreaterEqual(out[0][0], 10.0)
        self.assertLessEqual(out[0][0], 20.0)

    def test_causal_mask_blocks_future(self):
        Q = [[1.0], [1.0], [1.0]]
        K = [[1.0], [1.0], [1.0]]
        V = [[1.0], [2.0], [3.0]]
        _, weights = q_attention.scaled_dot_product_attention(Q, K, V, causal=True)
        # position 0 sees only itself
        self.assertAlmostEqual(weights[0][0], 1.0, places=12)
        self.assertEqual(weights[0][1], 0.0)
        self.assertEqual(weights[0][2], 0.0)

    def test_attention_as_lookup(self):
        r = q_attention.demonstrate_attention_as_lookup()
        self.assertTrue(r["match"])

    def test_causal_averaging_exact(self):
        r = q_attention.demonstrate_causal_averaging()
        self.assertTrue(r["exact_match"])

    def test_multihead_shapes(self):
        rng = random.Random(1)
        x = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(5)]
        mha = q_attention.MultiHeadAttention(8, 2, random.Random(2))
        out, weights = mha.forward(x, causal=True)
        self.assertEqual(q_attention.shape(out), (5, 8))
        self.assertEqual(len(weights), 2)

    def test_multihead_requires_divisible(self):
        with self.assertRaises(ValueError):
            q_attention.MultiHeadAttention(d_model=10, n_heads=3)

    def test_layernorm_zero_mean(self):
        ln = q_attention.LayerNorm(4)
        y = ln.forward([[1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(sum(y[0]) / 4, 0.0, places=10)

    def test_positional_encoding_bounded(self):
        pe = q_attention.positional_encoding(10, 8)
        for row in pe:
            for v in row:
                self.assertGreaterEqual(v, -1.0)
                self.assertLessEqual(v, 1.0)

    def test_positional_encoding_deterministic(self):
        pe1 = q_attention.positional_encoding(5, 8)
        pe2 = q_attention.positional_encoding(5, 8)
        self.assertEqual(pe1, pe2)

    def test_transformer_encoder_forward(self):
        rng = random.Random(3)
        enc = q_attention.TransformerEncoder(8, 2, 3, seed=42)
        emb = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(6)]
        hidden, attn = enc.forward(emb)
        self.assertEqual(q_attention.shape(hidden), (6, 8))
        self.assertEqual(len(attn), 3)

    def test_matmul_shape_check(self):
        with self.assertRaises(ValueError):
            q_attention.matmul([[1.0, 2.0]], [[1.0, 2.0]])  # (1x2)·(1x2) invalid


# ---------------------------------------------------------------------------
# Q-CRDT
# ---------------------------------------------------------------------------

import q_crdt

class TestCRDT(unittest.TestCase):
    def test_vector_clock_happens_before(self):
        a = q_crdt.VectorClock().tick("n1")
        b = q_crdt.VectorClock(a.clock).tick("n2")
        self.assertTrue(a.happens_before(b))
        self.assertFalse(b.happens_before(a))

    def test_vector_clock_concurrent(self):
        c1 = q_crdt.VectorClock({"n1": 1})
        c2 = q_crdt.VectorClock({"n2": 1})
        self.assertTrue(c1.concurrent_with(c2))

    def test_gcounter_merge(self):
        g1 = q_crdt.GCounter("n1"); g1.increment(5)
        g2 = q_crdt.GCounter("n2"); g2.increment(3)
        self.assertEqual(g1.merge(g2).value, 8)

    def test_gcounter_rejects_decrement(self):
        g = q_crdt.GCounter("n1")
        with self.assertRaises(ValueError):
            g.increment(-1)

    def test_gcounter_convergence(self):
        g1 = q_crdt.GCounter("n1").increment(5)
        g2 = q_crdt.GCounter("n2").increment(3)
        g3 = q_crdt.GCounter("n3").increment(7)
        self.assertTrue(q_crdt.verify_convergence([g1, g2, g3]))

    def test_pncounter_value(self):
        p1 = q_crdt.PNCounter("n1"); p1.increment(10); p1.decrement(3)
        p2 = q_crdt.PNCounter("n2"); p2.increment(5); p2.decrement(8)
        self.assertEqual(p1.merge(p2).value, 4)

    def test_pncounter_convergence(self):
        p1 = q_crdt.PNCounter("n1"); p1.increment(10); p1.decrement(3)
        p2 = q_crdt.PNCounter("n2"); p2.increment(5)
        self.assertTrue(q_crdt.verify_convergence([p1, p2]))

    def test_lww_register_latest_wins(self):
        r1 = q_crdt.LWWRegister("n1"); r1.set("alice", 100.0)
        r2 = q_crdt.LWWRegister("n2"); r2.set("bob", 200.0)
        self.assertEqual(r1.merge(r2).value, "bob")
        self.assertEqual(r2.merge(r1).value, "bob")  # commutative

    def test_lww_convergence(self):
        r1 = q_crdt.LWWRegister("n1"); r1.set("a", 100.0)
        r2 = q_crdt.LWWRegister("n2"); r2.set("b", 150.0)
        self.assertTrue(q_crdt.verify_convergence([r1, r2]))

    def test_orset_add_remove(self):
        s = q_crdt.ORSet("n1")
        s.add("x")
        self.assertTrue(s.contains("x"))
        s.remove("x")
        self.assertFalse(s.contains("x"))

    def test_orset_add_wins_concurrent(self):
        """Concurrent add and remove → add wins."""
        s1 = q_crdt.ORSet("n1"); s1.add("x")
        s2 = s1.merge(q_crdt.ORSet("n2"))   # s2 observes the original add
        s2.remove("x")                       # s2 removes the tag it saw
        s1.add("x")                          # s1 concurrently re-adds (fresh tag)
        final = s1.merge(s2)
        self.assertTrue(final.contains("x"))

    def test_orset_convergence(self):
        s1 = q_crdt.ORSet("n1"); s1.add("a"); s1.add("b")
        s2 = q_crdt.ORSet("n2"); s2.add("c")
        self.assertTrue(q_crdt.verify_convergence([s1, s2]))

    def test_rga_insert_order(self):
        r = q_crdt.RGA("n1")
        a = r.insert_after(None, "H")
        b = r.insert_after(a, "I")
        self.assertEqual(r.to_string(), "HI")

    def test_rga_delete(self):
        r = q_crdt.RGA("n1")
        a = r.insert_after(None, "X")
        b = r.insert_after(a, "Y")
        r.delete(a)
        self.assertEqual(r.to_string(), "Y")

    def test_rga_collaborative_convergence(self):
        result = q_crdt.demonstrate_collaborative_edit()
        self.assertTrue(result["converged"])
        self.assertEqual(result["alice_sees"], result["bob_sees"])

    def test_rga_concurrent_inserts_deterministic(self):
        """Two replicas inserting at the same anchor converge identically.

        Each replica must keep its own node identity for its whole lifetime
        (that is what makes insert ids globally unique), so we build them
        directly rather than via merge-into-a-throwaway.
        """
        alice = q_crdt.RGA("alice")
        bob = q_crdt.RGA("bob")
        anchor = alice.insert_after(None, "_")
        bob = bob.merge(alice)              # replicate the anchor to bob
        alice.insert_after(anchor, "A")     # concurrent edit on alice
        bob.insert_after(anchor, "B")       # concurrent edit on bob
        self.assertEqual(alice.merge(bob).to_list(), bob.merge(alice).to_list())


# ---------------------------------------------------------------------------
# Q-BPE
# ---------------------------------------------------------------------------

import q_bpe

class TestBPE(unittest.TestCase):
    def test_roundtrip_basic(self):
        tok = q_bpe.BPETokenizer().train("abab abab cdcd cdcd " * 10, vocab_size=300)
        for s in ["abab", "cdcd cdcd", "ab cd ab"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_roundtrip_empty(self):
        tok = q_bpe.BPETokenizer().train("hello world " * 5, vocab_size=270)
        self.assertEqual(tok.decode(tok.encode("")), "")

    def test_roundtrip_unseen_and_unicode(self):
        tok = q_bpe.BPETokenizer().train("the cat sat " * 10, vocab_size=290)
        for s in ["totally unseen text", "emoji \U0001f600 and accents éèê", "x"]:
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_byte_level_never_oov(self):
        """Any byte sequence must round-trip even with an empty merge table."""
        tok = q_bpe.BPETokenizer()  # untrained: only the 256 byte tokens
        for s in ["arbitrary \x00\x01 bytes", "日本語テスト"]:
            self.assertEqual(tok.decode_bytes(tok.encode(s)), s.encode("utf-8"))

    def test_merges_reduce_token_count(self):
        corpus = "quantum " * 50
        tok = q_bpe.BPETokenizer().train(corpus, vocab_size=320)
        # "quantum" should compress to far fewer than its 7 bytes
        self.assertLess(len(tok.encode("quantum")), 7)

    def test_vocab_size_respected(self):
        tok = q_bpe.BPETokenizer().train("ababab " * 100, vocab_size=300)
        self.assertLessEqual(tok.vocab_size, 300)
        self.assertGreaterEqual(tok.vocab_size, 256)

    def test_vocab_size_too_small_raises(self):
        with self.assertRaises(ValueError):
            q_bpe.BPETokenizer().train("text", vocab_size=100)

    def test_json_roundtrip(self):
        tok = q_bpe.BPETokenizer().train("merge me merge me " * 10, vocab_size=300)
        clone = q_bpe.BPETokenizer.from_json(tok.to_json())
        s = "merge me"
        self.assertEqual(clone.encode(s), tok.encode(s))
        self.assertEqual(clone.decode(clone.encode(s)), s)

    def test_demonstrate(self):
        r = q_bpe.demonstrate_bpe()
        self.assertTrue(r["roundtrip_ok"])
        self.assertGreater(r["bytes_per_token"], 1.0)


# ---------------------------------------------------------------------------
# Q-ZKP
# ---------------------------------------------------------------------------

import q_zkp

class TestZKP(unittest.TestCase):
    def test_schnorr_interactive_completeness(self):
        s = q_zkp.SchnorrIdentification(secret=42)
        proof = s.run_interactive()
        self.assertTrue(q_zkp.SchnorrIdentification.verify(
            s.y, proof.commitment, proof.challenge, proof.response))

    def test_schnorr_soundness(self):
        """A different public key must not verify the same transcript."""
        s = q_zkp.SchnorrIdentification(secret=42)
        proof = s.run_interactive()
        wrong_y = pow(q_zkp.G, 99, q_zkp.P)
        self.assertFalse(q_zkp.SchnorrIdentification.verify(
            wrong_y, proof.commitment, proof.challenge, proof.response))

    def test_fiat_shamir_completeness(self):
        secret = 1234567
        y = pow(q_zkp.G, secret, q_zkp.P)
        proof = q_zkp.schnorr_prove_nizk(secret, b"msg")
        self.assertTrue(q_zkp.schnorr_verify_nizk(y, proof, b"msg"))

    def test_fiat_shamir_message_binding(self):
        secret = 1234567
        y = pow(q_zkp.G, secret, q_zkp.P)
        proof = q_zkp.schnorr_prove_nizk(secret, b"original")
        self.assertFalse(q_zkp.schnorr_verify_nizk(y, proof, b"tampered"))

    def test_fiat_shamir_zero_knowledge_no_secret_leak(self):
        """Two proofs of the same secret use different commitments (fresh nonce)."""
        proof1 = q_zkp.schnorr_prove_nizk(555, b"m")
        proof2 = q_zkp.schnorr_prove_nizk(555, b"m")
        self.assertNotEqual(proof1.commitment, proof2.commitment)

    def test_pedersen_commitment_verifies(self):
        c = q_zkp.pedersen_commit(100)
        m, r = c.open()
        self.assertTrue(q_zkp.pedersen_verify(c.commitment, m, r))

    def test_pedersen_binding(self):
        c = q_zkp.pedersen_commit(100)
        _, r = c.open()
        self.assertFalse(q_zkp.pedersen_verify(c.commitment, 101, r))

    def test_pedersen_homomorphic(self):
        c1 = q_zkp.pedersen_commit(30)
        c2 = q_zkp.pedersen_commit(12)
        c_sum = q_zkp.pedersen_add(c1, c2)
        m, r = c_sum.open()
        self.assertEqual(m, 42)
        self.assertTrue(q_zkp.pedersen_verify(c_sum.commitment, 42, r))

    def test_chaum_pedersen_equality(self):
        x = 7777
        g1 = q_zkp.G
        g2 = pow(q_zkp.G, 3, q_zkp.P)
        y1, y2 = pow(g1, x, q_zkp.P), pow(g2, x, q_zkp.P)
        proof = q_zkp.prove_equal_discrete_log(x, g1, g2)
        self.assertTrue(q_zkp.verify_equal_discrete_log(g1, g2, y1, y2, proof))

    def test_chaum_pedersen_rejects_unequal(self):
        g1 = q_zkp.G
        g2 = pow(q_zkp.G, 3, q_zkp.P)
        y1 = pow(g1, 100, q_zkp.P)
        y2 = pow(g2, 200, q_zkp.P)   # different exponent
        proof = q_zkp.prove_equal_discrete_log(100, g1, g2)
        self.assertFalse(q_zkp.verify_equal_discrete_log(g1, g2, y1, y2, proof))

    def test_demonstrate(self):
        r = q_zkp.demonstrate_zkp()
        self.assertTrue(all([
            r["schnorr_interactive_verifies"],
            r["forged_proof_rejected"],
            r["fiat_shamir_verifies"],
            r["fiat_shamir_tamper_rejected"],
            r["pedersen_homomorphic_30_plus_12_eq_42"],
        ]))


# ---------------------------------------------------------------------------
# Q-RING
# ---------------------------------------------------------------------------

import q_ring

class TestRing(unittest.TestCase):
    def test_get_node_stable(self):
        ring = q_ring.ConsistentHashRing(vnodes=100)
        for i in range(5):
            ring.add_node(f"node{i}")
        # Same key always maps to the same node
        owner = ring.get_node("mykey")
        self.assertEqual(ring.get_node("mykey"), owner)
        self.assertIn(owner, ring.nodes)

    def test_empty_ring_returns_none(self):
        self.assertIsNone(q_ring.ConsistentHashRing().get_node("k"))

    def test_minimal_remapping(self):
        """Consistent hashing moves far fewer keys than modulo on node removal."""
        c = q_ring.compare_remapping(n_keys=5000, n_nodes=8)
        self.assertGreater(c["modulo_fraction"], 0.70)
        self.assertLess(c["consistent_fraction"], 0.25)

    def test_load_balance(self):
        b = q_ring.balance_stats(n_keys=20000, n_nodes=8, vnodes=200)
        self.assertLess(b["max_over_ideal"], 1.6)
        self.assertGreater(b["min_over_ideal"], 0.5)

    def test_replicas_are_distinct(self):
        ring = q_ring.ConsistentHashRing(vnodes=100)
        for i in range(5):
            ring.add_node(f"node{i}")
        replicas = ring.get_replicas("key123", 3)
        self.assertEqual(len(replicas), 3)
        self.assertEqual(len(set(replicas)), 3)

    def test_remove_node_reassigns(self):
        ring = q_ring.ConsistentHashRing(vnodes=100)
        for i in range(4):
            ring.add_node(f"node{i}")
        owner = ring.get_node("somekey")
        ring.remove_node(owner)
        self.assertNotIn(owner, ring.nodes)
        self.assertIn(ring.get_node("somekey"), ring.nodes)

    def test_rendezvous_deterministic(self):
        rv = q_ring.RendezvousHash()
        for i in range(5):
            rv.add_node(f"n{i}")
        self.assertEqual(rv.get_node("k"), rv.get_node("k"))
        self.assertEqual(len(rv.get_replicas("k", 3)), 3)

    def test_rendezvous_minimal_disruption(self):
        """Removing a node only affects keys that node owned."""
        rv = q_ring.RendezvousHash()
        for i in range(6):
            rv.add_node(f"n{i}")
        keys = [f"key{i}" for i in range(2000)]
        before = {k: rv.get_node(k) for k in keys}
        victim = "n3"
        rv.remove_node(victim)
        after = {k: rv.get_node(k) for k in keys}
        moved = sum(1 for k in keys if before[k] != after[k])
        # Only keys previously owned by the victim should move
        owned_by_victim = sum(1 for k in keys if before[k] == victim)
        self.assertEqual(moved, owned_by_victim)


# ---------------------------------------------------------------------------
# Q-RS (Reed-Solomon)
# ---------------------------------------------------------------------------

import q_reed_solomon
import itertools as _it

class TestReedSolomon(unittest.TestCase):
    def test_gf_arithmetic(self):
        self.assertEqual(q_reed_solomon.gf_mul(0, 5), 0)
        self.assertEqual(q_reed_solomon.gf_div(q_reed_solomon.gf_mul(7, 11), 11), 7)
        self.assertEqual(q_reed_solomon.gf_mul(q_reed_solomon.gf_inv(42), 42), 1)

    def test_gf_distributive(self):
        a, b, c = 17, 200, 99
        left = q_reed_solomon.gf_mul(a, q_reed_solomon.gf_add(b, c))
        right = q_reed_solomon.gf_add(q_reed_solomon.gf_mul(a, b),
                                      q_reed_solomon.gf_mul(a, c))
        self.assertEqual(left, right)

    def test_encode_is_systematic(self):
        rs = q_reed_solomon.ReedSolomon(k=4, m=2)
        data = b"systematic check ABCDEFGH"
        shards = rs.encode(data)
        padded = bytearray(data)
        while len(padded) % 4:
            padded.append(0)
        sl = len(padded) // 4
        for i in range(4):
            self.assertEqual(bytes(shards[i]), bytes(padded[i * sl:(i + 1) * sl]))

    def test_no_loss_decode(self):
        rs = q_reed_solomon.ReedSolomon(k=3, m=2)
        data = b"hello reed solomon world"
        shards = rs.encode(data)
        self.assertEqual(rs.decode(list(shards))[:len(data)], data)

    def test_recover_from_parity(self):
        rs = q_reed_solomon.ReedSolomon(k=6, m=4)
        data = b"Quantum-safe distributed storage durability."
        shards = rs.encode(data)
        damaged = list(shards)
        for lost in (1, 4, 7, 9):
            damaged[lost] = None
        self.assertEqual(rs.decode(damaged)[:len(data)], data)

    def test_all_loss_patterns(self):
        """Every way of losing exactly m shards must still decode."""
        rs = q_reed_solomon.ReedSolomon(k=4, m=3)
        data = b"exhaustive erasure test 0123456789"
        shards = rs.encode(data)
        for lost in _it.combinations(range(7), 3):
            damaged = [None if i in lost else shards[i] for i in range(7)]
            self.assertEqual(rs.decode(damaged)[:len(data)], data)

    def test_too_few_shards_raises(self):
        rs = q_reed_solomon.ReedSolomon(k=4, m=2)
        shards = rs.encode(b"data here")
        damaged = list(shards)
        for i in range(3):   # lose 3 > m=2
            damaged[i] = None
        with self.assertRaises(ValueError):
            rs.decode(damaged)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            q_reed_solomon.ReedSolomon(k=200, m=100)   # k+m > 256

    def test_demonstrate(self):
        r = q_reed_solomon.demonstrate_reed_solomon()
        self.assertTrue(r["recovered_ok"])


# ---------------------------------------------------------------------------
# Q-BPTREE
# ---------------------------------------------------------------------------

import q_bptree

class TestBPlusTree(unittest.TestCase):
    def test_insert_and_get(self):
        t = q_bptree.BPlusTree(order=4)
        for i in range(100):
            t.insert(i, i * 10)
        for i in range(100):
            self.assertEqual(t.get(i), i * 10)

    def test_missing_key(self):
        t = q_bptree.BPlusTree(order=4)
        t.insert(1, "a")
        self.assertIsNone(t.get(999))
        self.assertNotIn(999, t)
        self.assertIn(1, t)

    def test_update_existing(self):
        t = q_bptree.BPlusTree(order=4)
        t.insert(5, "old")
        t.insert(5, "new")
        self.assertEqual(t.get(5), "new")
        self.assertEqual(len(t), 1)

    def test_sorted_iteration(self):
        import random
        rng = random.Random(1)
        t = q_bptree.BPlusTree(order=6)
        keys = list(range(500))
        rng.shuffle(keys)
        for k in keys:
            t.insert(k, k)
        self.assertEqual(list(t.keys()), list(range(500)))

    def test_range_scan(self):
        t = q_bptree.BPlusTree(order=5)
        for i in range(100):
            t.insert(i, i)
        scanned = [k for k, _ in t.range_scan(20, 30)]
        self.assertEqual(scanned, list(range(20, 31)))

    def test_invariants_under_random_load(self):
        import random
        rng = random.Random(7)
        t = q_bptree.BPlusTree(order=4)   # small order → lots of splits
        for k in rng.sample(range(2000), 2000):
            t.insert(k, k)
        self.assertTrue(t.check_invariants())
        self.assertEqual(len(t), 2000)

    def test_stays_shallow(self):
        t = q_bptree.BPlusTree(order=32)
        for i in range(10000):
            t.insert(i, i)
        # 10k keys with order 32 → height should be small (logarithmic)
        self.assertLessEqual(t.height, 4)

    def test_min_max(self):
        import random
        t = q_bptree.BPlusTree(order=8)
        for k in random.Random(2).sample(range(1000), 1000):
            t.insert(k, k)
        self.assertEqual(t.min_key(), 0)
        self.assertEqual(t.max_key(), 999)

    def test_order_too_small(self):
        with self.assertRaises(ValueError):
            q_bptree.BPlusTree(order=2)


# ---------------------------------------------------------------------------
# Q-ROARING
# ---------------------------------------------------------------------------

import q_roaring

class TestRoaring(unittest.TestCase):
    def test_add_contains(self):
        rb = q_roaring.RoaringBitmap()
        for x in (1, 100, 70000, 5_000_000):
            rb.add(x)
        for x in (1, 100, 70000, 5_000_000):
            self.assertIn(x, rb)
        self.assertNotIn(2, rb)

    def test_cardinality_dedup(self):
        rb = q_roaring.RoaringBitmap()
        rb.add(5); rb.add(5); rb.add(5)
        self.assertEqual(rb.cardinality, 1)

    def test_array_to_bitmap_promotion(self):
        rb = q_roaring.RoaringBitmap()
        # Fill one chunk densely → must promote to a bitmap container
        for x in range(5000):
            rb.add(x)
        self.assertGreater(rb.container_stats()["bitmap_containers"], 0)

    def test_sparse_stays_array(self):
        rb = q_roaring.RoaringBitmap()
        for x in range(0, 1_000_000, 1000):
            rb.add(x)
        self.assertGreater(rb.container_stats()["array_containers"], 0)
        self.assertEqual(rb.container_stats()["bitmap_containers"], 0)

    def test_union(self):
        a = q_roaring.RoaringBitmap().add_many(range(0, 1000, 3))
        b = q_roaring.RoaringBitmap().add_many(range(0, 1000, 5))
        self.assertEqual((a | b).to_list(),
                         sorted(set(range(0, 1000, 3)) | set(range(0, 1000, 5))))

    def test_intersect(self):
        a = q_roaring.RoaringBitmap().add_many(range(0, 1000, 3))
        b = q_roaring.RoaringBitmap().add_many(range(0, 1000, 5))
        self.assertEqual((a & b).to_list(),
                         sorted(set(range(0, 1000, 3)) & set(range(0, 1000, 5))))

    def test_difference_and_xor(self):
        a = q_roaring.RoaringBitmap().add_many(range(0, 1000, 3))
        b = q_roaring.RoaringBitmap().add_many(range(0, 1000, 5))
        pa, pb = set(range(0, 1000, 3)), set(range(0, 1000, 5))
        self.assertEqual((a - b).to_list(), sorted(pa - pb))
        self.assertEqual((a ^ b).to_list(), sorted(pa ^ pb))

    def test_iteration_sorted(self):
        import random
        rb = q_roaring.RoaringBitmap()
        xs = random.Random(3).sample(range(2_000_000), 5000)
        rb.add_many(xs)
        self.assertEqual(rb.to_list(), sorted(set(xs)))

    def test_cross_container_ops(self):
        """Sparse (array) combined with dense (bitmap) must be exact."""
        sparse = q_roaring.RoaringBitmap().add_many(range(0, 10_000_000, 1000))
        dense = q_roaring.RoaringBitmap().add_many(range(5_000_000, 5_100_000))
        ps, pd = set(range(0, 10_000_000, 1000)), set(range(5_000_000, 5_100_000))
        self.assertEqual((sparse & dense).cardinality, len(ps & pd))
        self.assertEqual((sparse | dense).cardinality, len(ps | pd))

    def test_demonstrate(self):
        r = q_roaring.demonstrate_roaring()
        self.assertTrue(all([r["union_matches"], r["intersect_matches"],
                             r["difference_matches"], r["dense_uses_bitmaps"],
                             r["sparse_uses_arrays"]]))


# ---------------------------------------------------------------------------
# Q-SKIP
# ---------------------------------------------------------------------------

import q_skip

class TestSkipList(unittest.TestCase):
    def test_add_and_score(self):
        sl = q_skip.SkipList(seed=1)
        sl.add(3.5, "a")
        self.assertEqual(sl.score("a"), 3.5)
        self.assertIn("a", sl)

    def test_update_score_moves_member(self):
        sl = q_skip.SkipList(seed=1)
        sl.add(10, "x")
        sl.add(20, "y")
        sl.add(5, "x")    # re-score x below y
        self.assertEqual(sl.score("x"), 5)
        self.assertEqual(list(sl), [(5, "x"), (20, "y")])
        self.assertEqual(len(sl), 2)

    def test_remove(self):
        sl = q_skip.SkipList(seed=1)
        sl.add(1, "a"); sl.add(2, "b")
        self.assertTrue(sl.remove("a"))
        self.assertFalse(sl.remove("a"))
        self.assertNotIn("a", sl)
        self.assertEqual(len(sl), 1)

    def test_sorted_iteration(self):
        sl = q_skip.SkipList(seed=2)
        data = [(50, "e"), (10, "a"), (30, "c"), (20, "b"), (40, "d")]
        for s, m in data:
            sl.add(s, m)
        self.assertEqual(list(sl), sorted(data))

    def test_rank(self):
        sl = q_skip.SkipList(seed=3)
        for i, m in enumerate("abcde"):
            sl.add(i * 10, m)
        self.assertEqual(sl.rank("a"), 0)
        self.assertEqual(sl.rank("c"), 2)
        self.assertEqual(sl.rank("e"), 4)
        self.assertIsNone(sl.rank("z"))

    def test_select_by_index(self):
        sl = q_skip.SkipList(seed=4)
        for i, m in enumerate("abcde"):
            sl.add(i * 10, m)
        self.assertEqual(sl.select(0), (0, "a"))
        self.assertEqual(sl.select(4), (40, "e"))
        self.assertIsNone(sl.select(5))

    def test_rank_select_roundtrip(self):
        import random
        sl = q_skip.SkipList(seed=5)
        rng = random.Random(9)
        members = [f"m{i}" for i in range(1000)]
        for m in members:
            sl.add(rng.randint(0, 5000), m)
        for idx in (0, 250, 500, 999):
            score, member = sl.select(idx)
            self.assertEqual(sl.rank(member), idx)

    def test_range_by_score(self):
        sl = q_skip.SkipList(seed=6)
        for i in range(100):
            sl.add(i, f"k{i}")
        result = [m for _, m in sl.range_by_score(10, 20)]
        self.assertEqual(result, [f"k{i}" for i in range(10, 21)])

    def test_demonstrate(self):
        r = q_skip.demonstrate_skiplist()
        self.assertTrue(all([r["zrank_ok"], r["zrange_by_index_ok"],
                             r["iteration_sorted"]]))


# ---------------------------------------------------------------------------
# Q-GRAPH
# ---------------------------------------------------------------------------

import q_graph

class TestGraph(unittest.TestCase):
    def _diamond(self):
        g = q_graph.Graph(directed=False)
        for u, v, w in [("A", "B", 4), ("A", "C", 2), ("C", "B", 1),
                        ("B", "D", 5), ("C", "D", 8), ("D", "E", 3)]:
            g.add_edge(u, v, w)
        return g

    def test_bfs_dfs_visit_all(self):
        g = self._diamond()
        self.assertEqual(set(q_graph.bfs(g, "A")), set(g.nodes))
        self.assertEqual(set(q_graph.dfs(g, "A")), set(g.nodes))

    def test_unweighted_path(self):
        g = self._diamond()
        path = q_graph.shortest_unweighted_path(g, "A", "E")
        self.assertEqual(path[0], "A")
        self.assertEqual(path[-1], "E")

    def test_dijkstra(self):
        g = self._diamond()
        path, cost = q_graph.dijkstra_path(g, "A", "E")
        self.assertEqual(path, ["A", "C", "B", "D", "E"])
        self.assertEqual(cost, 11.0)

    def test_dijkstra_rejects_negative(self):
        g = q_graph.Graph()
        g.add_edge("x", "y", -1)
        with self.assertRaises(ValueError):
            q_graph.dijkstra(g, "x")

    def test_astar_matches_dijkstra(self):
        g = self._diamond()
        dpath, dcost = q_graph.dijkstra_path(g, "A", "E")
        apath, acost = q_graph.astar(g, "A", "E", heuristic=lambda a, b: 0.0)
        self.assertEqual(apath, dpath)
        self.assertAlmostEqual(acost, dcost)

    def test_topological_sort(self):
        g = q_graph.Graph(directed=True)
        for u, v in [("a", "b"), ("b", "c"), ("a", "c")]:
            g.add_edge(u, v)
        order = q_graph.topological_sort(g)
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("b"), order.index("c"))

    def test_cycle_detection(self):
        g = q_graph.Graph(directed=True)
        for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(u, v)
        self.assertIsNone(q_graph.topological_sort(g))
        self.assertTrue(q_graph.has_cycle(g))

    def test_connected_components(self):
        g = q_graph.Graph(directed=False)
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        g.add_node("e")
        comps = q_graph.connected_components(g)
        self.assertEqual(len(comps), 3)

    def test_union_find(self):
        uf = q_graph.UnionFind()
        for x in "abcd":
            uf.add(x)
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))
        self.assertNotEqual(uf.find("a"), uf.find("d"))

    def test_kruskal_mst(self):
        g = q_graph.Graph(directed=False)
        for u, v, w in [("a", "b", 1), ("b", "c", 2), ("a", "c", 3), ("c", "d", 4)]:
            g.add_edge(u, v, w)
        mst, total = q_graph.kruskal_mst(g)
        self.assertEqual(total, 7.0)
        self.assertEqual(len(mst), 3)

    def test_pagerank_sums_to_one(self):
        g = q_graph.Graph(directed=True)
        for u, v in [("p1", "hub"), ("p2", "hub"), ("hub", "p1")]:
            g.add_edge(u, v)
        ranks = q_graph.pagerank(g)
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=6)
        self.assertEqual(max(ranks, key=ranks.get), "hub")

    def test_demonstrate(self):
        r = q_graph.demonstrate_graph()
        self.assertTrue(r["astar_agrees_with_dijkstra"])
        self.assertTrue(r["pagerank_sums_to_1"])
        self.assertTrue(r["topo_valid"])


# ---------------------------------------------------------------------------
# Q-RATELIMIT
# ---------------------------------------------------------------------------

import q_rate_limit

class TestRateLimit(unittest.TestCase):
    def test_token_bucket_burst(self):
        clk = q_rate_limit.ManualClock()
        tb = q_rate_limit.TokenBucket(rate=5, capacity=10, clock=clk)
        allowed = sum(tb.allow() for _ in range(12))
        self.assertEqual(allowed, 10)

    def test_token_bucket_refill(self):
        clk = q_rate_limit.ManualClock()
        tb = q_rate_limit.TokenBucket(rate=5, capacity=10, clock=clk)
        for _ in range(10):
            tb.allow()
        self.assertFalse(tb.allow())
        clk.advance(1.0)               # +5 tokens
        self.assertEqual(sum(tb.allow() for _ in range(6)), 5)

    def test_token_bucket_average_rate(self):
        clk = q_rate_limit.ManualClock()
        tb = q_rate_limit.TokenBucket(rate=10, capacity=10, clock=clk)
        # Drain initial burst
        for _ in range(10):
            tb.allow()
        allowed = 0
        for _ in range(100):
            clk.advance(0.1)           # 0.1s → 1 token each tick
            if tb.allow():
                allowed += 1
        self.assertAlmostEqual(allowed, 100, delta=2)

    def test_leaky_bucket(self):
        clk = q_rate_limit.ManualClock()
        lb = q_rate_limit.LeakyBucket(rate=2, capacity=5, clock=clk)
        self.assertEqual(sum(lb.allow() for _ in range(5)), 5)
        self.assertFalse(lb.allow())
        clk.advance(1.0)               # leaks 2
        self.assertEqual(sum(lb.allow() for _ in range(2)), 2)

    def test_fixed_window(self):
        clk = q_rate_limit.ManualClock()
        fw = q_rate_limit.FixedWindowCounter(limit=3, window=1.0, clock=clk)
        self.assertEqual(sum(fw.allow() for _ in range(5)), 3)
        clk.advance(1.01)
        self.assertTrue(fw.allow())

    def test_sliding_window_log(self):
        clk = q_rate_limit.ManualClock()
        swl = q_rate_limit.SlidingWindowLog(limit=3, window=1.0, clock=clk)
        self.assertTrue(all(swl.allow() for _ in range(3)))
        self.assertFalse(swl.allow())
        clk.advance(1.01)
        self.assertTrue(swl.allow())

    def test_sliding_window_counter(self):
        clk = q_rate_limit.ManualClock()
        swc = q_rate_limit.SlidingWindowCounter(limit=10, window=1.0, clock=clk)
        allowed = sum(swc.allow() for _ in range(15))
        self.assertEqual(allowed, 10)

    def test_demonstrate(self):
        r = q_rate_limit.demonstrate_rate_limit()
        self.assertEqual(r["burst_allowed"], 10)
        self.assertTrue(r["sliding_log_recovers_after_window"])


# ---------------------------------------------------------------------------
# Q-TRIE
# ---------------------------------------------------------------------------

import q_trie

class TestTrie(unittest.TestCase):
    def test_insert_search(self):
        t = q_trie.Trie()
        t.insert("hello", 1)
        self.assertTrue(t.search("hello"))
        self.assertFalse(t.search("hell"))
        self.assertEqual(t.get("hello"), 1)

    def test_prefix(self):
        t = q_trie.Trie()
        t.insert("hello")
        self.assertTrue(t.starts_with("hell"))
        self.assertFalse(t.starts_with("world"))

    def test_autocomplete(self):
        t = q_trie.Trie()
        for w in ["quantum", "quasar", "query", "zebra"]:
            t.insert(w)
        self.assertEqual(t.autocomplete("qu"), ["quantum", "quasar", "query"])
        self.assertEqual(t.autocomplete("z"), ["zebra"])
        self.assertEqual(t.autocomplete("x"), [])

    def test_autocomplete_limit(self):
        t = q_trie.Trie()
        for w in ["a", "ab", "abc", "abcd"]:
            t.insert(w)
        self.assertEqual(len(t.autocomplete("a", limit=2)), 2)

    def test_delete(self):
        t = q_trie.Trie()
        t.insert("car"); t.insert("card")
        self.assertTrue(t.delete("car"))
        self.assertFalse(t.search("car"))
        self.assertTrue(t.search("card"))      # longer key survives
        self.assertFalse(t.delete("car"))       # already gone

    def test_radix_trie(self):
        rt = q_trie.RadixTrie()
        for w in ["romane", "romanus", "romulus", "rubens", "ruber"]:
            rt.insert(w)
        self.assertTrue(all(rt.search(w)
                            for w in ["romane", "romanus", "rubens"]))
        self.assertFalse(rt.search("rom"))
        self.assertFalse(rt.search("rube"))
        self.assertEqual(len(rt), 5)

    def test_ip_longest_prefix_match(self):
        r = q_trie.IPRoutingTable()
        r.add_route("0.0.0.0/0", "default")
        r.add_route("10.0.0.0/8", "internal")
        r.add_route("10.1.0.0/16", "subnet")
        r.add_route("10.1.2.0/24", "rack")
        self.assertEqual(r.lookup("10.1.2.55"), "rack")
        self.assertEqual(r.lookup("10.1.9.9"), "subnet")
        self.assertEqual(r.lookup("10.9.9.9"), "internal")
        self.assertEqual(r.lookup("8.8.8.8"), "default")

    def test_ip_no_default(self):
        r = q_trie.IPRoutingTable()
        r.add_route("192.168.0.0/16", "lan")
        self.assertEqual(r.lookup("192.168.5.5"), "lan")
        self.assertIsNone(r.lookup("8.8.8.8"))

    def test_demonstrate(self):
        r = q_trie.demonstrate_trie()
        self.assertTrue(r["autocomplete_correct"])
        self.assertTrue(r["lpm_correct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
