"""
Q-TRIE — Prefix trees & longest-prefix matching
  • Trie           — classic prefix tree: insert, search, prefix autocomplete
  • RadixTrie      — path-compressed trie (Patricia): fewer nodes, same API
  • IPRoutingTable — binary trie over IP bits for longest-prefix match (LPM)

Where these run in the real world:
  - autocomplete / typeahead, spell-checkers, dictionary lookups
  - IP routers: every packet's next hop is a longest-prefix match on the
    destination address against the routing table (the core of BGP/FIB lookups)
  - URL routers, T9 text entry, genome k-mer indexing

Zero dependencies.
"""

from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Classic trie
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "is_end", "value")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.is_end = False
        self.value: object = None


class Trie:
    """
    A character trie mapping string keys to optional values.

    Every node is one character; a path from the root spells a prefix, and
    nodes flagged `is_end` terminate a stored key. Lookup and insertion are
    O(len(key)) — independent of how many keys are stored.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def insert(self, key: str, value: object = None) -> None:
        node = self._root
        for ch in key:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _TrieNode()
                node.children[ch] = nxt
            node = nxt
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.value = value

    def search(self, key: str) -> bool:
        node = self._find(key)
        return node is not None and node.is_end

    def get(self, key: str) -> object:
        node = self._find(key)
        return node.value if (node and node.is_end) else None

    def __contains__(self, key: str) -> bool:
        return self.search(key)

    def starts_with(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def autocomplete(self, prefix: str, limit: Optional[int] = None) -> List[str]:
        """Return all stored keys beginning with `prefix`, sorted."""
        node = self._find(prefix)
        if node is None:
            return []
        out: List[str] = []
        self._collect(node, prefix, out, limit)
        return out

    def delete(self, key: str) -> bool:
        """Remove `key`. Returns True if it was present."""
        path: List[Tuple[_TrieNode, str]] = []
        node = self._root
        for ch in key:
            nxt = node.children.get(ch)
            if nxt is None:
                return False
            path.append((node, ch))
            node = nxt
        if not node.is_end:
            return False
        node.is_end = False
        node.value = None
        self._size -= 1
        # Prune now-empty leaf nodes bottom-up
        for parent, ch in reversed(path):
            child = parent.children[ch]
            if child.children or child.is_end:
                break
            del parent.children[ch]
        return True

    def keys(self) -> List[str]:
        out: List[str] = []
        self._collect(self._root, "", out, None)
        return out

    def _find(self, s: str) -> Optional[_TrieNode]:
        node = self._root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def _collect(self, node: _TrieNode, prefix: str,
                 out: List[str], limit: Optional[int]) -> None:
        if node.is_end:
            out.append(prefix)
            if limit is not None and len(out) >= limit:
                return
        for ch in sorted(node.children):
            if limit is not None and len(out) >= limit:
                return
            self._collect(node.children[ch], prefix + ch, out, limit)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"Trie(size={self._size})"


# ---------------------------------------------------------------------------
# Radix (Patricia) trie — path-compressed
# ---------------------------------------------------------------------------

class _RadixNode:
    __slots__ = ("edges", "is_end", "value")

    def __init__(self) -> None:
        # edge label (string) -> child node
        self.edges: Dict[str, "_RadixNode"] = {}
        self.is_end = False
        self.value: object = None


class RadixTrie:
    """
    A radix trie compresses chains of single-child nodes into one edge labelled
    with a substring, so storing 'romane'/'romanus'/'romulus' uses a handful of
    nodes instead of one per character. Same external behaviour as Trie, far
    fewer nodes for realistic key sets — what routers and IP stacks actually use.
    """

    def __init__(self) -> None:
        self._root = _RadixNode()
        self._size = 0

    @staticmethod
    def _common_prefix(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def insert(self, key: str, value: object = None) -> None:
        node = self._root
        rest = key
        while True:
            if rest == "":
                if not node.is_end:
                    self._size += 1
                node.is_end = True
                node.value = value
                return
            # Find an edge that shares a prefix with `rest`
            for label, child in list(node.edges.items()):
                cp = self._common_prefix(label, rest)
                if cp == 0:
                    continue
                if cp == len(label):
                    # Descend fully along this edge
                    node = child
                    rest = rest[cp:]
                    break
                # Split the edge at cp
                middle = _RadixNode()
                node.edges[label[:cp]] = middle
                del node.edges[label]
                middle.edges[label[cp:]] = child
                if cp == len(rest):
                    middle.is_end = True
                    middle.value = value
                    self._size += 1
                else:
                    leaf = _RadixNode()
                    leaf.is_end = True
                    leaf.value = value
                    middle.edges[rest[cp:]] = leaf
                    self._size += 1
                return
            else:
                # No sharing edge: add a fresh leaf
                leaf = _RadixNode()
                leaf.is_end = True
                leaf.value = value
                node.edges[rest] = leaf
                self._size += 1
                return

    def search(self, key: str) -> bool:
        node, rest = self._root, key
        while rest:
            for label, child in node.edges.items():
                if rest.startswith(label):
                    node = child
                    rest = rest[len(label):]
                    break
            else:
                return False
        return node.is_end

    def __contains__(self, key: str) -> bool:
        return self.search(key)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"RadixTrie(size={self._size})"


# ---------------------------------------------------------------------------
# IP routing table — longest-prefix match
# ---------------------------------------------------------------------------

class IPRoutingTable:
    """
    A binary trie over the 32 bits of an IPv4 address. Each route is a CIDR
    prefix (e.g. 10.0.0.0/8) mapped to a next hop. A lookup walks the address
    bit-by-bit and returns the next hop of the LONGEST matching prefix — exactly
    how a router's forwarding table (FIB) chooses where to send each packet.
    """

    def __init__(self) -> None:
        # node = [child0, child1, next_hop_or_None]
        self._root: list = [None, None, None]

    @staticmethod
    def _ip_to_int(ip: str) -> int:
        parts = ip.split(".")
        if len(parts) != 4:
            raise ValueError(f"invalid IPv4: {ip}")
        val = 0
        for p in parts:
            b = int(p)
            if not 0 <= b <= 255:
                raise ValueError(f"invalid octet in {ip}")
            val = (val << 8) | b
        return val

    def add_route(self, cidr: str, next_hop: str) -> None:
        """Add a route like '10.0.0.0/8' -> next_hop."""
        net, _, plen = cidr.partition("/")
        prefix_len = int(plen)
        addr = self._ip_to_int(net)
        node = self._root
        for i in range(prefix_len):
            bit = (addr >> (31 - i)) & 1
            if node[bit] is None:
                node[bit] = [None, None, None]
            node = node[bit]
        node[2] = next_hop

    def lookup(self, ip: str) -> Optional[str]:
        """Return the next hop for `ip` via longest-prefix match, or None."""
        addr = self._ip_to_int(ip)
        node = self._root
        best = node[2]   # a default route at / 0, if any
        for i in range(32):
            bit = (addr >> (31 - i)) & 1
            node = node[bit]
            if node is None:
                break
            if node[2] is not None:
                best = node[2]   # a longer prefix matched — prefer it
        return best

    def __repr__(self) -> str:
        return "IPRoutingTable()"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_trie() -> dict:
    """
    Autocomplete from a small dictionary, then resolve packet next-hops via
    longest-prefix match against a routing table — the two canonical trie uses.
    """
    # Autocomplete
    t = Trie()
    for word in ["quantum", "quasar", "query", "queue", "quick", "zebra"]:
        t.insert(word)
    suggestions = t.autocomplete("qu")

    # Radix trie equivalence on the same key set
    rt = RadixTrie()
    for word in ["quantum", "quasar", "query", "queue", "quick", "zebra"]:
        rt.insert(word)
    radix_ok = all(rt.search(w) for w in ["quantum", "zebra"]) and not rt.search("qu")

    # IP longest-prefix-match routing
    router = IPRoutingTable()
    router.add_route("0.0.0.0/0", "default-gateway")
    router.add_route("10.0.0.0/8", "internal")
    router.add_route("10.1.0.0/16", "subnet-1")
    router.add_route("10.1.2.0/24", "rack-7")

    routes = {
        "10.1.2.55": router.lookup("10.1.2.55"),   # most specific: rack-7
        "10.1.9.9":  router.lookup("10.1.9.9"),    # subnet-1
        "10.9.9.9":  router.lookup("10.9.9.9"),    # internal
        "8.8.8.8":   router.lookup("8.8.8.8"),     # default-gateway
    }

    return {
        "autocomplete_qu": suggestions,
        "autocomplete_correct": suggestions == ["quantum", "quasar", "query", "queue", "quick"],
        "radix_search_ok": radix_ok,
        "ip_routes": routes,
        "lpm_correct": routes == {
            "10.1.2.55": "rack-7",
            "10.1.9.9": "subnet-1",
            "10.9.9.9": "internal",
            "8.8.8.8": "default-gateway",
        },
    }
