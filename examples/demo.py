"""
Q-Drop demo — runs all 5 reference implementations end to end.

Usage (from the repository root):
    python3 examples/demo.py

No dependencies. If this script prints results, the bundle works on your machine.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q_hnsw import HNSWGraph
from q_lsm import LSMTree
from q_col import ColumnStore
from q_stream import StreamProcessor, Event
from q_mem import ArenaAllocator


def divider(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
divider("1. Q-HNSW — vector search (the algorithm behind Pinecone)")

random.seed(7)
index = HNSWGraph(dim=16, M=8)
docs = {i: [random.gauss(0, 1) for _ in range(16)] for i in range(500)}
for doc_id, vec in docs.items():
    index.insert(doc_id, vec)

query = docs[42]  # search for a known vector
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

print(f"Processed 50 sensor events over {t}ms into {len(closed)} windows:")
for w in closed[:5]:
    print(f"  window [{w.start:5d},{w.end:5d})  count={w.count:2d}  avg={w.value:.1f}")
print("  ...")

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

print(f"\n{'=' * 60}\n  All 5 systems ran successfully.\n{'=' * 60}")
