"""
Q-GRAPH — Core graph algorithms (the math behind search, social, routing)
  • BFS / DFS traversal and shortest unweighted paths
  • Dijkstra's algorithm        (Dijkstra 1959) — weighted shortest paths
  • A* search                   (Hart, Nilsson & Raphael 1968) — heuristic routing
  • Topological sort            (Kahn 1962) — dependency ordering + cycle detection
  • Union-Find / connected components (Tarjan)
  • PageRank                    (Page & Brin 1998) — Google's founding algorithm
  • Kruskal's minimum spanning tree (Kruskal 1956)

These power: Google Maps & GPS routing (Dijkstra/A*), build systems & package
managers (topological sort), social-network and citation ranking (PageRank),
network design and clustering (MST/union-find), and graph databases (Neo4j).

Zero dependencies — uses a binary heap built on heapq from the stdlib.
"""

from __future__ import annotations
import heapq
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Hashable, Iterator, List, Optional, Set, Tuple

Node = Hashable


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------

class Graph:
    """
    Weighted graph (directed or undirected) stored as an adjacency map:
        node -> {neighbor: weight}
    """

    def __init__(self, directed: bool = False):
        self.directed = directed
        self._adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)

    def add_node(self, u: Node) -> None:
        _ = self._adj[u]   # touch to create

    def add_edge(self, u: Node, v: Node, weight: float = 1.0) -> None:
        self._adj[u][v] = weight
        if not self.directed:
            self._adj[v][u] = weight
        else:
            _ = self._adj[v]   # ensure v exists as a node

    def neighbors(self, u: Node) -> Dict[Node, float]:
        return self._adj.get(u, {})

    @property
    def nodes(self) -> List[Node]:
        return list(self._adj.keys())

    def edges(self) -> Iterator[Tuple[Node, Node, float]]:
        seen: Set[Tuple[Node, Node]] = set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                if not self.directed and (v, u) in seen:
                    continue
                seen.add((u, v))
                yield u, v, w

    def __len__(self) -> int:
        return len(self._adj)

    def __repr__(self) -> str:
        n_edges = sum(len(d) for d in self._adj.values())
        if not self.directed:
            n_edges //= 2
        kind = "directed" if self.directed else "undirected"
        return f"Graph({kind}, nodes={len(self._adj)}, edges={n_edges})"


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def bfs(graph: Graph, start: Node) -> List[Node]:
    """Breadth-first traversal order from `start`."""
    visited: Set[Node] = {start}
    order: List[Node] = []
    q = deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order


def dfs(graph: Graph, start: Node) -> List[Node]:
    """Depth-first traversal order from `start` (iterative)."""
    visited: Set[Node] = set()
    order: List[Node] = []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        # Push neighbors in reverse for a natural left-to-right visit
        for v in reversed(list(graph.neighbors(u))):
            if v not in visited:
                stack.append(v)
    return order


def shortest_unweighted_path(graph: Graph, src: Node, dst: Node
                             ) -> Optional[List[Node]]:
    """BFS shortest path (fewest edges) from src to dst, or None."""
    if src == dst:
        return [src]
    prev: Dict[Node, Node] = {src: src}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in graph.neighbors(u):
            if v not in prev:
                prev[v] = u
                if v == dst:
                    return _reconstruct(prev, src, dst)
                q.append(v)
    return None


# ---------------------------------------------------------------------------
# Dijkstra / A*
# ---------------------------------------------------------------------------

def dijkstra(graph: Graph, src: Node) -> Tuple[Dict[Node, float], Dict[Node, Node]]:
    """
    Single-source shortest paths over non-negative edge weights.
    Returns (distance_map, predecessor_map).
    """
    dist: Dict[Node, float] = {src: 0.0}
    prev: Dict[Node, Node] = {}
    visited: Set[Node] = set()
    pq: List[Tuple[float, Node]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.neighbors(u).items():
            if w < 0:
                raise ValueError("Dijkstra requires non-negative weights")
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def dijkstra_path(graph: Graph, src: Node, dst: Node
                  ) -> Tuple[Optional[List[Node]], float]:
    """Shortest weighted path and its cost (path=None if unreachable)."""
    dist, prev = dijkstra(graph, src)
    if dst not in dist:
        return None, float("inf")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[dst]


def astar(graph: Graph, src: Node, dst: Node,
          heuristic: Callable[[Node, Node], float]
          ) -> Tuple[Optional[List[Node]], float]:
    """
    A* shortest path using an admissible `heuristic(node, goal)` estimate.
    With heuristic≡0 this reduces exactly to Dijkstra; a good heuristic
    explores far fewer nodes — the GPS-routing speedup.
    """
    g_score: Dict[Node, float] = {src: 0.0}
    prev: Dict[Node, Node] = {}
    pq: List[Tuple[float, Node]] = [(heuristic(src, dst), src)]
    visited: Set[Node] = set()

    while pq:
        _, u = heapq.heappop(pq)
        if u == dst:
            path = [dst]
            while path[-1] != src:
                path.append(prev[path[-1]])
            path.reverse()
            return path, g_score[dst]
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.neighbors(u).items():
            tentative = g_score[u] + w
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                prev[v] = u
                heapq.heappush(pq, (tentative + heuristic(v, dst), v))
    return None, float("inf")


# ---------------------------------------------------------------------------
# Topological sort (Kahn) + cycle detection
# ---------------------------------------------------------------------------

def topological_sort(graph: Graph) -> Optional[List[Node]]:
    """
    Kahn's algorithm: a linear order respecting all directed edges, or None
    if the graph has a cycle. The core of build systems and package managers.
    """
    if not graph.directed:
        raise ValueError("topological sort requires a directed graph")

    indeg: Dict[Node, int] = {u: 0 for u in graph.nodes}
    for u in graph.nodes:
        for v in graph.neighbors(u):
            indeg[v] = indeg.get(v, 0) + 1

    q = deque(sorted((u for u, d in indeg.items() if d == 0), key=str))
    order: List[Node] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return order if len(order) == len(graph.nodes) else None   # None ⇒ cycle


def has_cycle(graph: Graph) -> bool:
    if graph.directed:
        return topological_sort(graph) is None
    # Undirected: union-find — a within-component edge closes a cycle
    uf = UnionFind()
    for u in graph.nodes:
        uf.add(u)
    for u, v, _ in graph.edges():
        if uf.find(u) == uf.find(v):
            return True
        uf.union(u, v)
    return False


# ---------------------------------------------------------------------------
# Union-Find / connected components
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set with path compression + union by rank (near-O(1) ops)."""

    def __init__(self) -> None:
        self._parent: Dict[Node, Node] = {}
        self._rank: Dict[Node, int] = {}

    def add(self, x: Node) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: Node) -> Node:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:           # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Node, b: Node) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True


def connected_components(graph: Graph) -> List[Set[Node]]:
    """Group nodes into connected components (treats edges as undirected)."""
    uf = UnionFind()
    for u in graph.nodes:
        uf.add(u)
    for u, v, _ in graph.edges():
        uf.union(u, v)
    groups: Dict[Node, Set[Node]] = defaultdict(set)
    for u in graph.nodes:
        groups[uf.find(u)].add(u)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Minimum spanning tree (Kruskal)
# ---------------------------------------------------------------------------

def kruskal_mst(graph: Graph) -> Tuple[List[Tuple[Node, Node, float]], float]:
    """Minimum spanning tree edges and total weight (undirected graph)."""
    if graph.directed:
        raise ValueError("MST requires an undirected graph")
    uf = UnionFind()
    for u in graph.nodes:
        uf.add(u)
    mst: List[Tuple[Node, Node, float]] = []
    total = 0.0
    for u, v, w in sorted(graph.edges(), key=lambda e: e[2]):
        if uf.union(u, v):
            mst.append((u, v, w))
            total += w
    return mst, total


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

def pagerank(graph: Graph, damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-9) -> Dict[Node, float]:
    """
    PageRank via power iteration on the random-surfer model.

    rank(v) = (1-d)/N + d · Σ_{u→v} rank(u)/outdeg(u),  with dangling-node mass
    redistributed uniformly. Converges to the stationary distribution; ranks
    sum to 1. The algorithm that launched Google.
    """
    nodes = graph.nodes
    n = len(nodes)
    if n == 0:
        return {}
    rank = {u: 1.0 / n for u in nodes}
    out_deg = {u: len(graph.neighbors(u)) for u in nodes}

    for _ in range(max_iter):
        dangling = sum(rank[u] for u in nodes if out_deg[u] == 0)
        new_rank = {u: (1.0 - damping) / n + damping * dangling / n for u in nodes}
        for u in nodes:
            if out_deg[u] == 0:
                continue
            share = damping * rank[u] / out_deg[u]
            for v in graph.neighbors(u):
                new_rank[v] += share
        # Convergence check (L1)
        delta = sum(abs(new_rank[u] - rank[u]) for u in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reconstruct(prev: Dict[Node, Node], src: Node, dst: Node) -> List[Node]:
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_graph() -> dict:
    """
    Build a small road network + a tiny web graph and exercise the key
    algorithms: weighted shortest path, A* vs Dijkstra agreement, topological
    sort of a build dependency DAG, and PageRank on a link graph.
    """
    # Road network (undirected, weighted by distance)
    roads = Graph(directed=False)
    for u, v, w in [("A", "B", 4), ("A", "C", 2), ("C", "B", 1),
                    ("B", "D", 5), ("C", "D", 8), ("D", "E", 3)]:
        roads.add_edge(u, v, w)
    path, cost = dijkstra_path(roads, "A", "E")
    # A* with a zero heuristic must agree with Dijkstra
    apath, acost = astar(roads, "A", "E", heuristic=lambda a, b: 0.0)

    # Build-dependency DAG → topological order
    build = Graph(directed=True)
    for u, v in [("compile", "link"), ("link", "test"),
                 ("generate", "compile"), ("compile", "package"),
                 ("package", "test")]:
        build.add_edge(u, v)
    topo = topological_sort(build)

    # Web link graph → PageRank (a popular page pointed to by many)
    web = Graph(directed=True)
    for u, v in [("p1", "hub"), ("p2", "hub"), ("p3", "hub"),
                 ("hub", "p1"), ("p2", "p3")]:
        web.add_edge(u, v)
    ranks = pagerank(web)
    top_page = max(ranks, key=ranks.get)

    return {
        "dijkstra_path": path,
        "dijkstra_cost": cost,
        "astar_agrees_with_dijkstra": apath == path and abs(acost - cost) < 1e-9,
        "topological_order": topo,
        "topo_valid": topo is not None,
        "pagerank_top_page": top_page,
        "pagerank_sums_to_1": abs(sum(ranks.values()) - 1.0) < 1e-6,
        "components": len(connected_components(roads)),
    }
