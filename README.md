# Q-Drop — Infrastructure Algorithms, Implemented For Real

**Five working, tested reference implementations of the core algorithms behind
modern data infrastructure** — the same algorithms that power RocksDB, Pinecone,
Kafka Streams, DuckDB, and jemalloc.

Every line of code is in this repository. Read it all before you pay anything.

## What's in the bundle

| Module | What it implements | Powers (in production systems) |
|--------|--------------------|-------------------------------|
| [`src/q_hnsw.py`](src/q_hnsw.py) | HNSW approximate nearest-neighbor graph (Malkov & Yashunin 2018) | Pinecone, Weaviate, Qdrant, pgvector |
| [`src/q_lsm.py`](src/q_lsm.py) | Log-Structured Merge-tree with WAL, MemTable, SSTables, compaction | RocksDB, LevelDB, Cassandra |
| [`src/q_col.py`](src/q_col.py) | Columnar engine: bitmap filters, hash GROUP BY, dictionary & RLE encoding | DuckDB, Arrow, Polars, Parquet |
| [`src/q_stream.py`](src/q_stream.py) | Windowed stream processing: tumbling/sliding/session windows, watermarks | Kafka Streams, Flink, Dataflow |
| [`src/q_mem.py`](src/q_mem.py) | Arena allocator: size classes, slabs, free lists, fragmentation stats | jemalloc, APR pools, game engines |

Pure Python 3, zero dependencies, every file heavily commented for learning.

## Try it right now

```bash
git clone https://github.com/AmineCHABANE/Q-Drop-Marketplace
cd Q-Drop-Marketplace

# Run the test suite (42 tests)
python3 -m unittest discover -s src/tests -v

# Run the end-to-end demo of all 5 systems
python3 examples/demo.py
```

If those two commands work on your machine, the product works. That's the proof.

## Who this is for

- **Engineers preparing system-design / infra interviews** — these five algorithms
  come up constantly, and reading a clean 300-line implementation beats reading
  a 300,000-line production codebase.
- **Students & self-learners** — each file explains the *why* behind every data
  structure, with references to the original papers.
- **Developers building their own tools** — use these as the starting skeleton
  for your own storage engine, vector index, or stream processor.

## What is NOT claimed

These are **reference implementations optimized for clarity, not speed**.
They are not benchmarked against, and will not outperform, RocksDB, FAISS,
or Flink. If you need production performance, use those projects — and use
this code to understand what they're doing under the hood.

## License & pricing

The code is **source-visible**: free to read, run, and use for personal
study and evaluation.

**Commercial use** (using the code or derived code in a product or at work)
requires a one-time license: **€9.99**, lifetime, covering all current and
future modules in this repository. See [LICENSE](LICENSE).

- Pay by card (Stripe): https://buy.stripe.com/9B6eV60Gg90Z0TbcZk9fW01
- PayPal: Aminechabane7@gmail.com (€9.99, "Q-Drop License")

30-day money-back guarantee, no questions asked: email
Aminechabane7@gmail.com.

## Website

https://aminechabane.github.io/Q-Drop-Marketplace/
