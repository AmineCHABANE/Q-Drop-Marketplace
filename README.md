# Q-Drop — Infrastructure + Quantum Algorithms, Implemented For Real

**Ten working, tested reference implementations** — the core algorithms behind
modern data infrastructure *and* the quantum algorithms that will reshape
computing over the next decade.

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

### Quantum (simulation)

| Module | What it implements | Why it matters |
|--------|--------------------|----------------|
| [`src/q_grover.py`](src/q_grover.py) | Grover's search: full statevector, amplitude amplification, QFT | O(√N) quantum speedup over classical O(N) search |
| [`src/q_kyber.py`](src/q_kyber.py) | CRYSTALS-Kyber KEM (NIST FIPS 203): Module-LWE keygen/encap/decap | Replacing RSA/ECC in TLS, SSH, Signal — quantum-safe |
| [`src/q_shor.py`](src/q_shor.py) | Shor's algorithm: QFT period-finding, classical GCD post-processing | Why RSA breaks with quantum computers — O(n³) vs exp |
| [`src/q_qec.py`](src/q_qec.py) | QEC: bit/phase-flip codes, Shor [[9,1,3]], Steane [[7,1,3]] stabilizer | Fault-tolerant QC — the engineering layer enabling quantum advantage |
| [`src/q_vqe.py`](src/q_vqe.py) | VQE + QAOA: parametric ansatz, parameter-shift gradients, H₂ molecule | NISQ quantum chemistry and combinatorial optimization today |

Pure Python 3, zero external dependencies, every file commented with
references to original papers.

## Try it right now

```bash
git clone https://github.com/AmineCHABANE/Q-Drop-Marketplace
cd Q-Drop-Marketplace

# Run the full test suite (87 tests)
python3 -m unittest discover -s src/tests -v

# Run the end-to-end demo of all 10 systems
python3 examples/demo.py
```

If those two commands work on your machine, the product works. That's the proof.

## Who this is for

- **Engineers preparing interviews** — HNSW, LSM-trees, columnar storage, and
  stream processing are standard system-design topics; Grover's, Shor's, and
  QEC are appearing in "quantum readiness" conversations at FAANG.
- **Developers building on classical infra** — use these as the starting skeleton
  for your own storage engine, vector index, or stream processor.
- **Anyone getting quantum-ready** — Kyber (the post-quantum TLS standard) and
  VQE (running on real NISQ hardware today) are not future tech — they're shipping now.
- **Students & researchers** — each file explains the *why* with references to
  the original papers (Grover 1996, Shor 1997, Peruzzo 2014, NIST FIPS 203).

## What is NOT claimed

These are **reference implementations optimized for clarity, not speed**.
They are not benchmarked against, and will not outperform, FAISS, RocksDB,
Qiskit, or PennyLane. Use this code to understand what those tools do under
the hood — then deploy the battle-hardened version in production.

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
