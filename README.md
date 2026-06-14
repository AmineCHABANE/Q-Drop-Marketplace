# Q-Drop — Infrastructure + Quantum + AI Algorithms, Implemented For Real

**Eighteen working, tested reference implementations** — the core algorithms
behind modern data infrastructure, the AI mechanism inside every large language
model, the quantum algorithms that will reshape computing, the quantum-safe
formats for everyday data in that future, and the distributed-systems primitives
that underpin it all.

Every line of code is in this repository. Read it all before you pay anything.

## What's in the bundle

### Infrastructure (classical)

| Module | What it implements | Powers in production |
|--------|--------------------|----------------------|
| [`src/q_hnsw.py`](src/q_hnsw.py) | HNSW approximate nearest-neighbor graph (Malkov & Yashunin 2018) | Pinecone, Weaviate, Qdrant, pgvector |
| [`src/q_lsm.py`](src/q_lsm.py) | Log-Structured Merge-tree: WAL, MemTable, SSTables, compaction | RocksDB, LevelDB, Cassandra |
| [`src/q_col.py`](src/q_col.py) | Columnar engine: bitmap filters, hash GROUP BY, dict & RLE encoding | DuckDB, Arrow, Polars, Parquet |
| [`src/q_stream.py`](src/q_stream.py) | Windowed stream processing: tumbling/sliding/session, watermarks | Kafka Streams, Flink, Dataflow |
| [`src/q_mem.py`](src/q_mem.py) | Arena allocator: size classes, slabs, free lists, fragmentation stats | jemalloc, APR pools, game engines |
| [`src/q_raft.py`](src/q_raft.py) | **Raft consensus**: leader election, log replication, snapshots, partition tolerance | etcd, CockroachDB, TiKV, Consul |
| [`src/q_bloom.py`](src/q_bloom.py) | **Bloom + Xor + Cuckoo filters**, HyperLogLog, MinHash / LSH | RocksDB, Redis, Elasticsearch, Cassandra |
| [`src/q_crdt.py`](src/q_crdt.py) | **CRDTs**: vector clocks, G/PN counters, LWW register, OR-set, RGA sequence | Figma, Linear, Notion, Automerge, Yjs |

### AI (the mechanism inside every LLM)

| Module | What it implements | Why it matters |
|--------|--------------------|----------------|
| [`src/q_attention.py`](src/q_attention.py) | **The Transformer from scratch**: scaled dot-product + multi-head attention, causal masking, positional encoding, LayerNorm, GELU FFN, full encoder block | The attention mechanism (Vaswani et al. 2017) inside GPT, Claude, Gemini, Llama, BERT, Stable Diffusion |

### Quantum (simulation)

| Module | What it implements | Why it matters |
|--------|--------------------|----------------|
| [`src/q_grover.py`](src/q_grover.py) | Grover's search: full statevector, amplitude amplification, QFT | O(√N) quantum speedup over classical O(N) search |
| [`src/q_kyber.py`](src/q_kyber.py) | CRYSTALS-Kyber KEM (NIST FIPS 203): Module-LWE keygen/encap/decap | Replacing RSA/ECC in TLS, SSH, Signal — quantum-safe |
| [`src/q_dilithium.py`](src/q_dilithium.py) | **CRYSTALS-Dilithium signatures (NIST FIPS 204)**: Module-LWE sign/verify | Replacing ECDSA/RSA-PSS in TLS certificates, SSH, code signing |
| [`src/q_shor.py`](src/q_shor.py) | Shor's algorithm: QFT period-finding, classical GCD post-processing | Why RSA breaks with quantum computers — O(n³) vs exp |
| [`src/q_qec.py`](src/q_qec.py) | QEC: bit/phase-flip codes, Shor [[9,1,3]], Steane [[7,1,3]] stabilizer | Fault-tolerant QC — the engineering layer enabling quantum advantage |
| [`src/q_vqe.py`](src/q_vqe.py) | VQE + QAOA: parametric ansatz, parameter-shift gradients, H₂ molecule | NISQ quantum chemistry and combinatorial optimization today |

### Quantum-safe formats (the encodings of tomorrow)

| Module | What it implements | Why it matters |
|--------|--------------------|----------------|
| [`src/q_pack.py`](src/q_pack.py) | LZSS + canonical Huffman compression (the DEFLATE recipe, RFC 1951) | Post-quantum crypto is BIG — compression matters more, not less |
| [`src/q_sign.py`](src/q_sign.py) | Hash-based signatures: WOTS + Merkle tree (XMSS-style, cf. NIST FIPS 205) | Signatures that survive Shor's algorithm — security = SHA-256 only |
| [`src/q_qsf.py`](src/q_qsf.py) | **QSF** — Quantum-Safe File format: compression + Kyber encryption + hash signing in one container | What PGP/age look like when every primitive must survive a quantum computer |

Pure Python 3, zero external dependencies, every file commented with
references to the original papers.

### What Kyber + Dilithium together mean

Kyber and Dilithium are the two NIST-standardised post-quantum primitives.
Kyber handles *key encapsulation* (replace RSA/ECDH in TLS handshakes).
Dilithium handles *signatures* (replace ECDSA/RSA-PSS in certificates and
code signing). Together they make a complete quantum-safe replacement for
the cryptography that secures the entire internet today. Both are now live
in production (OpenSSH 9.x uses Kyber; Chrome 116+ negotiates Kyber KEM in
TLS 1.3). Dilithium is in FIPS 204, signed into law August 2024.

### Post-quantum license system (dogfooding)

License keys for this product are themselves issued with `q_sign` — WOTS
hash-based signatures verifiable offline against the Merkle root in
[`LICENSE_ROOT.txt`](LICENSE_ROOT.txt). No license server, no database,
quantum-unforgeable. Buyers verify with one command:

```bash
python3 tools/license_manager.py verify --key "<your key>"
```

Full flow documented in [LICENSING.md](LICENSING.md).

## Try it right now

```bash
git clone https://github.com/AmineCHABANE/Q-Drop-Marketplace
cd Q-Drop-Marketplace

# Run the full test suite (176 tests)
python3 -m unittest discover -s src/tests -v

# Run the end-to-end demo of all 18 systems
python3 examples/demo.py
```

If those two commands run without errors, every implementation in this bundle
works on your machine. That's the proof.

## Who this is for

- **Engineers preparing interviews** — HNSW, LSM-trees, columnar storage, Raft
  consensus, and CRDTs are standard system-design topics; the Transformer is the
  #1 ML-systems question; Grover's, Shor's, and QEC appear in "quantum readiness"
  conversations at FAANG.
- **Developers building on classical infra** — use these as the starting skeleton
  for your own storage engine, vector index, stream processor, or collaborative editor.
- **ML engineers** — `q_attention.py` is the entire Transformer mechanism in pure
  Python: read 400 commented lines instead of tracing a 100k-line framework.
- **Anyone getting quantum-ready** — Kyber & Dilithium (the post-quantum TLS
  standards) and VQE (running on real NISQ hardware today) are not future tech —
  they're shipping now.
- **Students & researchers** — each file explains the *why* with references to
  the original papers (Vaswani 2017, Shapiro 2011, Grover 1996, Shor 1997,
  NIST FIPS 203/204).

## What is NOT claimed

These are **reference implementations optimized for clarity, not speed**.
They are not benchmarked against, and will not outperform, FAISS, RocksDB,
PyTorch, Qiskit, or PennyLane. The Transformer here does a forward pass with
fixed/random weights — it is not a trained model and does no backprop. Use this
code to understand what those tools do under the hood — then deploy the
battle-hardened version in production.

## License & pricing

The code is **source-visible**: free to read, run, and use for personal
study, education, and evaluation.

**Commercial use** (using the code or derived code in a product or at work)
requires a one-time license: **€9.99**, lifetime, covering all current and
future modules in this repository. See [LICENSE](LICENSE).

- Pay by card (Stripe): https://buy.stripe.com/9B6eV60Gg90Z0TbcZk9fW01
- PayPal: Aminechabane7@gmail.com (€9.99, "Q-Drop License")

30-day money-back guarantee, no questions asked:
email Aminechabane7@gmail.com.

## Website

https://aminechabane.github.io/Q-Drop-Marketplace/
