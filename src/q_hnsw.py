"""
Q-HNSW: Hierarchical Navigable Small World Graph
Reference implementation of the HNSW approximate nearest neighbor algorithm.

Paper: Malkov & Yashunin (2018) "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs"
https://arxiv.org/abs/1603.09320

This is a clean, heavily commented Python implementation for studying and
understanding how vector databases (Pinecone, Weaviate, Qdrant) work internally.
Not optimized for production (use hnswlib/faiss for that) — optimized for clarity.

Usage:
    from q_hnsw import HNSWGraph

    index = HNSWGraph(dim=128, M=16, distance='cosine')
    index.insert(0, vector_0)
    index.insert(1, vector_1)
    results = index.search(query_vector, k=10)  # [(dist, id), ...]
"""

import math
import heapq
import random
from typing import Dict, List, Optional, Tuple


class HNSWGraph:
    """
    HNSW graph for approximate nearest neighbor (ANN) search.

    The key idea: build a multi-layer proximity graph where:
    - Upper layers are sparse long-range connections (highway network)
    - Lower layers are dense short-range connections (local search)

    Insert:
      1. Randomly assign node to a level (exponential distribution)
      2. From top level to node's level: greedy descent to find entry point
      3. From node's level to 0: search layer and connect to M nearest neighbors

    Search:
      1. Greedy descent from top layer to layer 1 (ef=1, just find entry point)
      2. Beam search at layer 0 with ef candidates
      3. Return top-k from final candidate set
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        distance: str = "cosine",
    ):
        """
        dim              : dimensionality of all vectors
        M                : max bidirectional links per node per layer
                           (higher M → better recall, more RAM, slower insert)
        ef_construction  : beam width during graph construction
                           (higher → better graph quality, slower inserts)
        distance         : 'cosine' | 'l2'
        """
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # layer-0 allows twice as many connections
        self.ef_construction = ef_construction

        if distance == "cosine":
            self._dist = self._cosine_dist
        elif distance == "l2":
            self._dist = self._l2_dist
        else:
            raise ValueError(f"distance must be 'cosine' or 'l2', got '{distance}'")

        # node_id → float list (stored as plain Python list for portability)
        self.vectors: Dict[int, List[float]] = {}

        # layers[level][node_id] = list of neighbor node_ids at that level
        self.graph: List[Dict[int, List[int]]] = []

        self.entry_point: Optional[int] = None
        self.max_level: int = -1

        # Normalization factor for level sampling: controls expected max level
        # E[max_level] ≈ log(n) / log(M) — same as the paper
        self._level_norm = 1.0 / math.log(max(M, 2))

    # ------------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_dist(a: List[float], b: List[float]) -> float:
        """
        Cosine distance = 1 - cosine_similarity.
        Range: [0, 2]. 0 means identical direction, 2 means opposite.
        """
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)

    @staticmethod
    def _l2_dist(a: List[float], b: List[float]) -> float:
        """Squared Euclidean distance (no sqrt, preserves ordering)."""
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_level(self) -> int:
        """
        Sample a level for a new node using exponential distribution.
        P(level >= k) = (1/M)^k, so most nodes are in layer 0.
        """
        return int(-math.log(random.random()) * self._level_norm)

    def _search_layer(
        self,
        query: List[float],
        entry_id: int,
        ef: int,
        level: int,
    ) -> List[Tuple[float, int]]:
        """
        Greedy beam search within a single graph layer.

        Uses two heaps:
        - `candidates`: min-heap, nodes to explore (nearest first)
        - `found`:      max-heap of size ef, best nodes found so far

        Loop: pop nearest candidate; if it's farther than worst-found, stop.
        Otherwise explore its neighbors.

        Returns sorted list of (distance, node_id), size ≤ ef.
        """
        if level >= len(self.graph) or entry_id not in self.graph[level]:
            # Layer doesn't exist yet or entry not connected — just return entry
            return [(self._dist(query, self.vectors[entry_id]), entry_id)]

        visited = {entry_id}
        entry_dist = self._dist(query, self.vectors[entry_id])

        # Min-heap for exploration: (dist, id)
        candidates = [(entry_dist, entry_id)]

        # Max-heap for results: (-dist, id) so heappop gives worst
        found = [(-entry_dist, entry_id)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            # If nearest unexplored is farther than our worst result, we're done
            worst_found = -found[0][0]
            if c_dist > worst_found:
                break

            for neighbor_id in self.graph[level].get(c_id, []):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                n_dist = self._dist(query, self.vectors[neighbor_id])
                worst_found = -found[0][0]

                if n_dist < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor_id))
                    heapq.heappush(found, (-n_dist, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)  # remove worst

        # Convert max-heap to sorted ascending list
        return sorted(((-neg_d, nid) for neg_d, nid in found))

    def _prune_neighbors(
        self,
        node_vec: List[float],
        candidates: List[Tuple[float, int]],
        M_max: int,
    ) -> List[int]:
        """
        Simple neighbor selection: keep M_max nearest candidates.

        The paper describes a more complex heuristic (Algorithm 4) that
        prefers diverse neighbors — you can upgrade to that for better recall.
        """
        return [nid for _, nid in sorted(candidates)[:M_max]]

    def _ensure_level(self, level: int) -> None:
        """Expand graph list to include the given level."""
        while len(self.graph) <= level:
            self.graph.append({})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, node_id: int, vector: List[float]) -> None:
        """
        Insert a vector.

        node_id : unique integer identifier
        vector  : list/array of floats, length must equal self.dim
        """
        if len(vector) != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {len(vector)}")

        self.vectors[node_id] = list(vector)
        level = self._sample_level()
        self._ensure_level(level)

        # Initialize node's adjacency lists at all its levels
        for l in range(level + 1):
            if node_id not in self.graph[l]:
                self.graph[l][node_id] = []

        # First node: nothing to connect to
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        ep = self.entry_point

        # Phase 1: descend from max_level to level+1 with ef=1 (just find entry)
        for lc in range(self.max_level, level, -1):
            if lc < len(self.graph):
                results = self._search_layer(self.vectors[node_id], ep, ef=1, level=lc)
                if results:
                    ep = results[0][1]

        # Phase 2: from min(level, max_level) down to 0, insert with full ef
        for lc in range(min(level, self.max_level), -1, -1):
            M_at_level = self.M_max0 if lc == 0 else self.M
            self._ensure_level(lc)

            candidates = self._search_layer(
                self.vectors[node_id], ep, ef=self.ef_construction, level=lc
            )
            neighbors = self._prune_neighbors(self.vectors[node_id], candidates, M_at_level)

            # Connect new node → its neighbors
            self.graph[lc][node_id] = neighbors

            # Connect neighbors → new node (bidirectional), then prune if over limit
            for nb_id in neighbors:
                if nb_id not in self.graph[lc]:
                    self.graph[lc][nb_id] = []
                self.graph[lc][nb_id].append(node_id)

                if len(self.graph[lc][nb_id]) > M_at_level:
                    nb_vec = self.vectors[nb_id]
                    nb_candidates = [
                        (self._dist(nb_vec, self.vectors[x]), x)
                        for x in self.graph[lc][nb_id]
                    ]
                    self.graph[lc][nb_id] = self._prune_neighbors(
                        nb_vec, nb_candidates, M_at_level
                    )

            if candidates:
                ep = candidates[0][1]

        # Update global entry point if this node reaches a higher level
        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def search(
        self,
        query: List[float],
        k: int,
        ef: int = 50,
    ) -> List[Tuple[float, int]]:
        """
        Find the k approximate nearest neighbors of query.

        query : list/array of floats
        k     : number of results to return
        ef    : search beam width (ef ≥ k; higher → better recall, slower)

        Returns: [(distance, node_id), ...] sorted by distance ascending
        """
        if self.entry_point is None or not self.vectors:
            return []

        ef = max(k, ef)
        ep = self.entry_point

        # Descent: top layers → layer 1, ef=1
        for lc in range(self.max_level, 0, -1):
            results = self._search_layer(query, ep, ef=1, level=lc)
            if results:
                ep = results[0][1]

        # Full beam search at layer 0
        results = self._search_layer(query, ep, ef=ef, level=0)
        return results[:k]

    def __len__(self) -> int:
        return len(self.vectors)

    def stats(self) -> dict:
        """Return graph statistics for debugging."""
        total_edges = sum(
            sum(len(nb) for nb in level.values()) for level in self.graph
        )
        return {
            "num_vectors": len(self.vectors),
            "num_levels": len(self.graph),
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "total_edges": total_edges,
            "avg_degree_l0": (
                sum(len(nb) for nb in self.graph[0].values()) / max(1, len(self.graph[0]))
                if self.graph else 0.0
            ),
        }
