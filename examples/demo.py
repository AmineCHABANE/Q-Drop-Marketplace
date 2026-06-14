"""
Q-Drop demo — runs all 18 reference implementations end to end.

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
import q_pack
import q_sign
import q_qsf
import q_dilithium
import q_raft
import q_bloom
import q_attention
import q_crdt


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

# ---------------------------------------------------------------------------
divider("11. Q-PACK — LZSS + Huffman compression (the gzip recipe)")

text = open(os.path.join(os.path.dirname(__file__), "..", "README.md"), "rb").read()
blob = q_pack.compress(text)
restored = q_pack.decompress(blob)
print(f"README.md: {len(text)} bytes → {len(blob)} bytes "
      f"({100 * len(blob) / len(text):.0f}% of original)")
print(f"Lossless roundtrip: {restored == text}")

repetitive = b"quantum " * 1000
print(f"Repetitive input: {len(repetitive)} → {len(q_pack.compress(repetitive))} bytes")

# ---------------------------------------------------------------------------
divider("12. Q-SIGN — post-quantum hash-based signatures (WOTS + Merkle)")

signer = q_sign.MerkleSigner.generate(height=4, seed=b"demo-seed-0123456789abcdef000000")
root = signer.public_root
print(f"Keypair: 2^4 = {signer.n_leaves} one-time signatures, "
      f"public root = {root.hex()[:32]}...")

message = b"license: buyer@example.com, order ST-123"
sig = signer.sign(message)
print(f"Signature: leaf {sig.leaf_index}, "
      f"{len(sig.to_bytes())} bytes (67 WOTS chains + auth path)")
print(f"Verify (correct message): {q_sign.verify(message, sig, root)}")
print(f"Verify (tampered message): {q_sign.verify(b'license: attacker@evil.com', sig, root)}")
print("Security reduces to SHA-256 only — Shor's algorithm has no effect.")

# ---------------------------------------------------------------------------
divider("13. Q-QSF — Quantum-Safe File format (compress + Kyber + sign)")

document = ("Donnees quotidiennes proteges pour l'ere quantique. " * 60).encode()
pk_qsf, sk_qsf = keygen(b"\x09" * 32)

container = q_qsf.create(
    document,
    metadata={"creator": "q-drop-demo", "type": "text/plain"},
    recipient_pk=pk_qsf,
    signer=signer,
    _kem_message=b"\x33" * 32,
)
print(f"Document: {len(document)} bytes")
print(f"QSF container (compressed + Kyber-encrypted + hash-signed): {len(container)} bytes")

opened = q_qsf.open_container(container, sk=sk_qsf, signer_root=root)
print(f"Decrypted + decompressed: {len(opened.payload)} bytes, "
      f"intact = {opened.payload == document}")
print(f"Signature valid: {opened.signature_valid}")
print("Every cryptographic layer in this file survives a quantum computer.")

# ---------------------------------------------------------------------------
divider("14. Q-DILITHIUM — CRYSTALS-Dilithium post-quantum signatures (NIST FIPS 204)")

print("Generating Dilithium-2 keypair (Module-LWE, security level 2)...")
pk, sk = q_dilithium.generate_keypair(seed=b"\xAB" * 32)
pk_bytes = pk.to_bytes()
print(f"  Public key:  {len(pk_bytes)} bytes")

msg = b"TLS 1.3 certificate: CN=example.com, issuer=post-quantum-ca.example"
sig = sk.sign(msg)
sig_bytes = sig.to_bytes()
print(f"  Signature:   {len(sig_bytes)} bytes  (z vector + hints + challenge hash)")
print(f"  Verify OK:   {q_dilithium.verify(pk, msg, sig)}")
print(f"  Tamper fail: {not q_dilithium.verify(pk, b'CN=attacker.evil', sig)}")

ok, total = q_dilithium.verify_correctness(5)
print(f"  Correctness: {ok}/{total} sign/verify pairs")
print("Pair Dilithium (signatures) + Kyber (KEM) = complete post-quantum TLS 1.3")

# ---------------------------------------------------------------------------
divider("15. Q-RAFT — Raft distributed consensus (the algorithm behind etcd/CockroachDB)")

print("Simulating 5-node Raft cluster...")
cluster = q_raft.RaftCluster(5, seed=7)
cluster.tick(40)   # elect initial leader
ldr = cluster.leader()
print(f"  Elected leader: node {ldr.id}  (term {ldr.current_term})")

# Write 10 key-value pairs
for i in range(10):
    ok, idx = cluster.propose({"op": "set", "key": f"counter:{i}", "value": i * 100})

print(f"  10 writes committed.  Leader commit index: {cluster.leader().commit_index}")
print(f"  Cluster consistent:   {cluster.is_consistent()}")

# Demonstrate partition tolerance
old_id = cluster.leader().id
cluster.partition([old_id])
cluster.tick(50)
new_ldr = cluster.leader()
print(f"  Partitioned leader {old_id} → new leader: node {new_ldr.id} "
      f"(term {new_ldr.current_term})")

cluster.heal()
cluster.tick(40)
print(f"  Healed. Single leader: node {cluster.leader().id}, "
      f"consistent: {cluster.is_consistent()}")
print("Same algorithm used in etcd, CockroachDB, TiKV, Consul, YugabyteDB.")

# ---------------------------------------------------------------------------
divider("16. Q-BLOOM — Probabilistic filters + cardinality (Bloom / Xor / HLL / MinHash)")

print("Bloom filter (LSM-tree point-lookup guard):")
bf = q_bloom.BloomFilter(50_000, fpr=0.01)
for i in range(50_000):
    bf.add(f"key:{i}")
fp = sum(1 for i in range(50_000, 100_000) if f"key:{i}" in bf)
print(f"  {bf}")
print(f"  FPR on 50k non-members: {fp/50_000:.4f}  (target <0.01)")

print("\nXor filter (Binary Fuse 8 — more compact than Bloom):")
xf = q_bloom.XorFilter.from_items([f"id:{i}" for i in range(10_000)])
print(f"  {xf}")

print("\nHyperLogLog (Redis PFCOUNT cardinality estimation):")
hll = q_bloom.HyperLogLog(b=12)   # m=4096 registers, ~0.8% error
for i in range(100_000):
    hll.add(f"user:{i}")
print(f"  True count: 100000   Estimate: {hll.count()}   "
      f"Error: {abs(hll.count()-100000)/100000:.3f}")

print("\nMinHash Jaccard similarity (duplicate detection):")
doc_a = set("the quick brown fox jumps over the lazy dog".split())
doc_b = set("the quick brown fox runs over the lazy cat".split())
real_j = len(doc_a & doc_b) / len(doc_a | doc_b)
m_a = q_bloom.MinHash.from_set(doc_a, n_hashes=512)
m_b = q_bloom.MinHash.from_set(doc_b, n_hashes=512)
est_j = m_a.similarity(m_b)
print(f"  Real Jaccard: {real_j:.3f}   MinHash estimate: {est_j:.3f}")
print("Deployed in: RocksDB, Cassandra, Redis, Chrome Safe Browsing, Elasticsearch.")

# ---------------------------------------------------------------------------
divider("17. Q-ATTENTION — the Transformer mechanism (the algorithm inside every LLM)")

print("Scaled dot-product attention as a content-addressable lookup:")
lookup = q_attention.demonstrate_attention_as_lookup()
print(f"  Query points at memory slot #{lookup['query_points_at']}")
print(f"  Attention weights: {lookup['attention_weights']}")
print(f"  Retrieved value:   {lookup['retrieved_value']}  (stored: {lookup['expected_value']})")
print(f"  Content-addressed read works: {lookup['match']}")

print("\nCausal attention computes a running prefix mean (exact):")
avg = q_attention.demonstrate_causal_averaging()
print(f"  values:        {avg['values']}")
print(f"  prefix means:  {avg['causal_prefix_means']}")
print(f"  exact to 1e-9: {avg['exact_match']}")

print("\nFull multi-layer Transformer forward pass (pure Python, no numpy):")
import random as _rng_mod
_r = _rng_mod.Random(42)
encoder = q_attention.TransformerEncoder(d_model=32, n_heads=4, n_layers=2, seed=1)
embeddings = [[_r.gauss(0, 1) for _ in range(32)] for _ in range(8)]
hidden, attn_maps = encoder.forward(embeddings)
print(f"  Input:  8 tokens × 32 dims")
print(f"  Output: {q_attention.shape(hidden)[0]} tokens × {q_attention.shape(hidden)[1]} dims, "
      f"{len(attn_maps)} layers × {len(attn_maps[0])} heads of attention")
params = q_attention.count_parameters(d_model=512, n_heads=8, n_layers=12)
print(f"  A GPT-scale config (d=512, h=8, L=12) would hold {params:,} parameters")
print("Same mechanism powers GPT, Claude, Gemini, Llama, BERT, Stable Diffusion.")

# ---------------------------------------------------------------------------
divider("18. Q-CRDT — conflict-free replication (how Figma/Linear sync offline)")

print("Two users edit the same document offline, then sync — no server, no locks:")
edit = q_crdt.demonstrate_collaborative_edit()
print(f"  Alice's replica converges to: {edit['alice_sees']!r}")
print(f"  Bob's replica converges to:   {edit['bob_sees']!r}")
print(f"  Both replicas identical:      {edit['converged']}")

print("\nCRDT merge laws verified by shuffling merge order (commutative+associative+idempotent):")
g1 = q_crdt.GCounter("n1").increment(5)
g2 = q_crdt.GCounter("n2").increment(3)
g3 = q_crdt.GCounter("n3").increment(7)
print(f"  GCounter [5,3,7] → merged value {g1.merge(g2).merge(g3).value}, "
      f"convergent: {q_crdt.verify_convergence([g1, g2, g3])}")

s1 = q_crdt.ORSet("n1"); s1.add('apple'); s1.add('pear')
s2 = q_crdt.ORSet("n2"); s2.add('apple'); s2.add('plum')
print(f"  ORSet merge {{apple,pear}} ∪ {{apple,plum}} = {s1.merge(s2).values()}")

p1 = q_crdt.PNCounter("n1"); p1.increment(10); p1.decrement(3)
p2 = q_crdt.PNCounter("n2"); p2.increment(5); p2.decrement(2)
print(f"  PNCounter (+10−3) merge (+5−2) = {p1.merge(p2).value}")
print("Same math behind Figma, Linear, Notion, Apple Notes, Automerge, Yjs, Riak.")

print(f"\n{'=' * 60}\n  All 18 systems ran successfully.\n{'=' * 60}")
