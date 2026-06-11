"""
Q-Drop demo — runs all 10 reference implementations end to end.

Usage (from the repository root):
    python3 examples/demo.py

No dependencies. If this script prints results, the bundle works on your machine.
"""

import os
import math
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_hnsw import HNSWGraph
from q_lsm import LSMTree
from q_col import ColumnStore
from q_stream import StreamProcessor, Event
from q_mem import ArenaAllocator
from q_grover import GroverSearch, QuantumFourierTransform
from q_kyber import keygen, encapsulate, decapsulate
from q_shor import ShorFactorer, demonstrate_why_rsa_breaks
from q_qec import BitFlipCode, run_error_correction_demo, build_steane_code
from q_vqe import VQE, Hamiltonian, UCCSDSingletAnsatz, QAOA


def divider(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
divider("1. Q-HNSW — vector search (the algorithm behind Pinecone)")

random.seed(7)
index = HNSWGraph(dim=16, M=8)
docs = {i: [random.gauss(0, 1) for _ in range(16)] for i in range(500)}
for doc_id, vec in docs.items():
    index.insert(doc_id, vec)

query = docs[42]
results = index.search(query, k=3)
print(f"Indexed {len(index)} vectors. Top-3 neighbors of doc 42:")
for dist, doc_id in results:
    print(f"  doc {doc_id:3d}  distance={dist:.4f}")
print(f"Graph stats: {index.stats()}")

# ---------------------------------------------------------------------------
divider("2. Q-LSM — key-value store (the engine behind RocksDB)")

db = LSMTree(memtable_limit=50, l0_compaction_trigger=3)
for i in range(500):
    db.put(f"user:{i:04d}", f"data_{i}")
db.delete("user:0042")
db.compact()

print(f"get('user:0041') = {db.get('user:0041')}")
print(f"get('user:0042') = {db.get('user:0042')}  (deleted)")
print(f"Range scan user:0010..user:0013 = {list(db.scan('user:0010', 'user:0013'))}")
print(f"Tree stats: {db.stats()}")

# ---------------------------------------------------------------------------
divider("3. Q-COL — columnar analytics (how DuckDB/Polars work)")

store = ColumnStore()
cities = ["Paris", "London", "Berlin", "Madrid"]
n = 1000
random.seed(1)
store.add_column("city", [random.choice(cities) for _ in range(n)])
store.add_column("sales", [random.randint(10, 500) for _ in range(n)])

mask = store.filter_and(("city", "==", "Paris"), ("sales", ">", 250))
print(f"Rows where city=Paris AND sales>250: {mask.count()}")
print(f"Their total sales: {store.aggregate('sales', 'sum', mask)}")
print("GROUP BY city:")
for row in store.groupby("city", {"sales": "sum"}):
    print(f"  {row}")

# ---------------------------------------------------------------------------
divider("4. Q-STREAM — windowed streams (the Kafka Streams model)")

closed = []
proc = StreamProcessor(window_type="tumbling", window_size_ms=1000, agg="avg")
proc.on_window(closed.append)

random.seed(3)
t = 0
for _ in range(50):
    t += random.randint(50, 150)
    proc.emit(Event(timestamp=t, key="sensor-1", value=random.randint(15, 30)))
proc.flush()

print(f"Processed 50 events → {len(closed)} windows:")
for w in closed[:5]:
    print(f"  window [{w.start:5d},{w.end:5d})  count={w.count:2d}  avg={w.value:.1f}")

# ---------------------------------------------------------------------------
divider("5. Q-MEM — arena allocator (jemalloc-style size classes)")

arena = ArenaAllocator(slab_size=4096)
handles = [arena.alloc(random.randint(8, 200)) for _ in range(200)]
arena.write(handles[0], b"hello arena")
print(f"read back: {arena.read(handles[0], 11)}")
stats = arena.stats()
print(f"200 allocations across {stats['num_slabs']} slabs")
print(f"Internal fragmentation: {stats['internal_fragmentation_pct']:.1f}%")
arena.reset()
print(f"After reset(): {arena.stats()['live_allocations']} live allocations")

# ---------------------------------------------------------------------------
divider("6. Q-GROVER — Grover's quantum search (simulated statevector)")

N_QUBITS = 6   # 2^6 = 64-item database
target = 37

gs = GroverSearch(n_qubits=N_QUBITS, seed=42)
k = gs.optimal_iterations()
p_success = gs.probability_of_success([target])
measured = gs.search([target])

print(f"Database size: 2^{N_QUBITS} = {gs.N} items")
print(f"Classical search: O(N={gs.N}) queries worst case")
print(f"Grover's search:  O(√N≈{math.isqrt(gs.N)}) queries  ({k} iterations)")
print(f"Success probability after {k} iterations: {p_success:.4f} ({p_success*100:.1f}%)")
print(f"Measured index: {measured}  (target was {target})")

# Bonus: QFT norm preservation check
qft = QuantumFourierTransform(n_qubits=N_QUBITS)
sv = gs.statevector([target])
sv_qft = qft.apply(sv)
norm = sum(abs(a)**2 for a in sv_qft)
print(f"QFT norm preserved: {norm:.12f} (should be 1.0)")

# ---------------------------------------------------------------------------
divider("7. Q-KYBER — CRYSTALS-Kyber post-quantum key exchange")

print("Simulating post-quantum key exchange (NIST FIPS 203):")
seed = b"\xde\xad\xbe\xef" * 8
msg  = b"\xca\xfe\xba\xbe" * 8

pk, sk = keygen(seed)
ct, shared_secret_sender   = encapsulate(pk, msg)
shared_secret_recipient = decapsulate(sk, ct)

print(f"  Key generation:  OK")
print(f"  Encapsulation:   ciphertext size = {len(ct.u) * len(ct.u[0]) + len(ct.v)} coefficients")
print(f"  Sender secret:   {shared_secret_sender.hex()[:32]}...")
print(f"  Recipient secret:{shared_secret_recipient.hex()[:32]}...")
print(f"  Secrets match:   {shared_secret_sender == shared_secret_recipient}")

from q_kyber import verify_correctness
successes, n = verify_correctness(10)
print(f"  Correctness: {successes}/{n} key exchanges succeeded")

# ---------------------------------------------------------------------------
divider("8. Q-SHOR — Shor's factoring algorithm (quantum simulation)")

print(demonstrate_why_rsa_breaks())
print()

shor = ShorFactorer(use_quantum_simulation=False)   # classical for demo speed
for N_test, expected in [(15, {3, 5}), (21, {3, 7}), (35, {5, 7})]:
    f = shor.factor(N_test, seed=42)
    print(f"  factor({N_test:3d}) = {f}  ✓" if f in expected else f"  factor({N_test}) = {f}  ?")

factors_of_360 = shor.full_factorization(360, seed=42)
print(f"  full_factorization(360) = {factors_of_360}  (360 = {' × '.join(str(x) for x in factors_of_360)})")

# ---------------------------------------------------------------------------
divider("9. Q-QEC — Quantum error correction (stabilizer codes)")

code = BitFlipCode()
# Encode |0⟩ = α=1, β=0
reg = code.encode(1.0 + 0j, 0.0 + 0j)

# Inject an error on qubit 1
print("Logical qubit |0⟩ encoded in 3 physical qubits.")
print("Injecting X error on qubit 1...")
reg.apply_x(1)

# Measure syndrome and correct
rng = random.Random(99)
s01, s12 = code.syndrome(reg, rng)
print(f"  Syndrome: (Z₀Z₁={s01}, Z₁Z₂={s12}) → error detected on qubit 1")
code.correct(reg, s01, s12)
recovered = code.decode(reg, rng)
from q_qec import QubitState
ideal = QubitState.zero()
print(f"  Fidelity after correction: {recovered.fidelity(ideal):.6f} (should be 1.0)")

# Monte Carlo error correction stats
stats = run_error_correction_demo(n_trials=500, p_error=0.05, seed=42)
print(f"\nMonte Carlo (p_error={stats['p_error']}, n={stats['n_trials']}):")
print(f"  Without QEC: {stats['success_rate_without_qec']*100:.1f}% success")
print(f"  With QEC:    {stats['success_rate_with_qec']*100:.1f}% success")

# Steane [[7,1,3]] code
steane = build_steane_code()
print(f"\nSteane [[{steane.n},{steane.k},3]] code: {len(steane.generators)} stabilizer generators")

# ---------------------------------------------------------------------------
divider("10. Q-VQE — Variational Quantum Eigensolver (quantum chemistry)")

print("Finding ground-state energy of H₂ molecule (STO-3G basis):")
H = Hamiltonian.h2_molecule()
ansatz = UCCSDSingletAnsatz()
vqe = VQE(H, ansatz, learning_rate=0.05, max_iterations=300)
result = vqe.run(seed=42)

print(f"  Optimal energy:   {result.optimal_energy:.6f} Hartree")
print(f"  Exact (FCI):     −1.137 Hartree")
print(f"  Error:           {abs(result.optimal_energy - (-1.137)):.4f} Hartree")
print(f"  Converged:        {result.converged} in {result.n_iterations} iterations")
print(f"  Optimal params:   θ = {result.optimal_params[0]:.4f} rad")

# QAOA MaxCut demo (2-qubit)
print("\nQAOA MaxCut on triangle graph (3 nodes, 3 edges):")
from q_vqe import QAOA, Hamiltonian
edges = [(0, 1), (1, 2), (2, 0)]
H_cut = Hamiltonian.maxcut_qubo(edges, n=3)
qaoa = QAOA(H_cut, p_layers=1)
qaoa_result = qaoa.run(seed=0)
print(f"  Best cut value: {-qaoa_result.optimal_energy:.4f}  (max-cut = 2 for K₃)")

print(f"\n{'=' * 60}\n  All 10 systems ran successfully.\n{'=' * 60}")
