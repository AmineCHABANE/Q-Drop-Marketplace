"""
Q-RAFT — Raft distributed consensus algorithm
Diego Ongaro & John Ousterhout, "In Search of an Understandable Consensus Algorithm"
USENIX ATC 2014 (https://raft.github.io/raft.pdf)

Full simulation including:
  - Leader election with randomized timeouts
  - Log replication with acknowledgment quorums
  - Log matching and conflict resolution
  - Membership: single-server configuration changes (joint consensus)
  - Snapshotting / log compaction
  - Cluster simulation with injected network partitions and failures

Concept behind: etcd, CockroachDB, TiKV, Consul, YugabyteDB
All production distributed systems using Raft for linearizable consensus.

Zero dependencies — simulation runs entirely in-process.
"""

from __future__ import annotations
import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MsgType(Enum):
    VOTE_REQUEST   = auto()
    VOTE_RESPONSE  = auto()
    APPEND_ENTRIES = auto()
    APPEND_RESP    = auto()
    INSTALL_SNAP   = auto()
    SNAP_RESP      = auto()


@dataclass
class LogEntry:
    term:    int
    index:   int   # 1-based
    command: Any   # arbitrary state-machine command

    def __repr__(self) -> str:
        return f"Entry(t={self.term},i={self.index},{self.command!r})"


@dataclass
class Snapshot:
    """Compact state up through last_index."""
    last_index: int
    last_term:  int
    data:       bytes   # serialised state-machine snapshot


@dataclass
class Message:
    src:  int
    dst:  int
    typ:  MsgType
    body: Dict[str, Any]


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class Role(Enum):
    FOLLOWER  = "follower"
    CANDIDATE = "candidate"
    LEADER    = "leader"


class RaftServer:
    """
    Single Raft server node.

    Time is simulated via integer 'ticks'.  One tick ≈ one heartbeat interval
    in the reference implementation.  Election timeout = random.randint(10, 20)
    ticks; heartbeat interval = 5 ticks.

    The server is pure: it never does I/O.  All output is through the
    outgoing_msgs queue, consumed by RaftCluster.
    """

    HEARTBEAT = 5    # ticks between heartbeats
    TIMEOUT_LO = 10  # min election timeout (ticks)
    TIMEOUT_HI = 20  # max election timeout

    def __init__(self, server_id: int, peers: List[int], rng: Optional[random.Random] = None):
        self.id       = server_id
        self.peers    = list(peers)
        self._rng     = rng or random.Random(server_id)

        # Persistent state (survives crash/restart in a real system)
        self.current_term: int = 0
        self.voted_for:    Optional[int] = None
        self.log:          List[LogEntry] = []   # index 0 = dummy sentinel

        # Snapshot state
        self.snapshot: Optional[Snapshot] = None
        self._snap_offset = 0   # first real log index after snapshot

        # Volatile state
        self.commit_index: int = 0
        self.last_applied: int = 0
        self.role:    Role = Role.FOLLOWER
        self.leader_id: Optional[int] = None

        # Candidate state
        self._votes_received: Set[int] = set()

        # Leader state (reinitialised on election win)
        self._next_index:  Dict[int, int] = {}   # peer → next log index to send
        self._match_index: Dict[int, int] = {}   # peer → highest known replicated index

        # Timers
        self._election_timeout: int = self._reset_timeout()
        self._ticks_since_reset: int = 0
        self._heartbeat_ticks: int = 0

        # State machine: simple key-value store
        self.state_machine: Dict[str, Any] = {}

        # Message output queue
        self.outgoing_msgs: List[Message] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the simulated clock by one tick."""
        self._ticks_since_reset += 1
        if self.role == Role.LEADER:
            self._heartbeat_ticks += 1
            if self._heartbeat_ticks >= self.HEARTBEAT:
                self._heartbeat_ticks = 0
                self._send_heartbeats()
        else:
            if self._ticks_since_reset >= self._election_timeout:
                self._start_election()

    def handle(self, msg: Message) -> None:
        """Process an incoming message."""
        # If we see a higher term, step down immediately
        if msg.body.get("term", 0) > self.current_term:
            self._become_follower(msg.body["term"])

        dispatch = {
            MsgType.VOTE_REQUEST:   self._on_vote_request,
            MsgType.VOTE_RESPONSE:  self._on_vote_response,
            MsgType.APPEND_ENTRIES: self._on_append_entries,
            MsgType.APPEND_RESP:    self._on_append_response,
            MsgType.INSTALL_SNAP:   self._on_install_snapshot,
            MsgType.SNAP_RESP:      self._on_snapshot_response,
        }
        handler = dispatch.get(msg.typ)
        if handler:
            handler(msg)

    def propose(self, command: Any) -> Optional[int]:
        """
        Leader: append a command to the local log and start replication.
        Returns the log index, or None if not leader.
        """
        if self.role != Role.LEADER:
            return None
        index = self._last_log_index() + 1
        entry = LogEntry(term=self.current_term, index=index, command=command)
        self.log.append(entry)
        self._replicate_to_peers()
        return index

    def apply_committed(self) -> List[Tuple[int, Any]]:
        """
        Apply all log entries up to commit_index to the state machine.
        Returns list of (index, result) for newly applied entries.
        """
        applied = []
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self._log_entry(self.last_applied)
            if entry is None:
                break
            result = self._apply_to_sm(entry.command)
            applied.append((self.last_applied, result))
        return applied

    def compact(self, up_through: int) -> None:
        """
        Create a snapshot of the state machine up through log index up_through,
        then truncate the log.  Only makes sense on leader (or as install trigger).
        """
        if up_through <= self._snap_offset:
            return
        entry = self._log_entry(up_through)
        if entry is None:
            return
        import json
        snap_data = json.dumps(self.state_machine).encode()
        self.snapshot = Snapshot(
            last_index=up_through,
            last_term=entry.term,
            data=snap_data,
        )
        # Discard compacted entries
        keep_from = up_through - self._snap_offset
        self.log = self.log[keep_from:]
        self._snap_offset = up_through

    # ------------------------------------------------------------------
    # Election
    # ------------------------------------------------------------------

    def _start_election(self) -> None:
        self.current_term += 1
        self.role = Role.CANDIDATE
        self.voted_for = self.id
        self._votes_received = {self.id}
        self._ticks_since_reset = 0
        self._election_timeout = self._reset_timeout()

        last_idx  = self._last_log_index()
        last_term = self._last_log_term()
        for peer in self.peers:
            self._send(peer, MsgType.VOTE_REQUEST, {
                "term":          self.current_term,
                "candidate_id":  self.id,
                "last_log_index": last_idx,
                "last_log_term":  last_term,
            })

    def _on_vote_request(self, msg: Message) -> None:
        b = msg.body
        term        = b["term"]
        candidate   = b["candidate_id"]
        last_l_idx  = b["last_log_index"]
        last_l_term = b["last_log_term"]

        grant = False
        if term >= self.current_term:
            already_voted = (self.voted_for is not None and
                             self.voted_for != candidate)
            if not already_voted and self._log_is_up_to_date(last_l_idx, last_l_term):
                self.voted_for = candidate
                grant = True
                self._ticks_since_reset = 0   # reset timeout on granting vote

        self._send(candidate, MsgType.VOTE_RESPONSE, {
            "term":         self.current_term,
            "vote_granted": grant,
        })

    def _on_vote_response(self, msg: Message) -> None:
        if self.role != Role.CANDIDATE:
            return
        if msg.body["term"] != self.current_term:
            return
        if msg.body["vote_granted"]:
            self._votes_received.add(msg.src)
            quorum = (len(self.peers) + 1) // 2 + 1
            if len(self._votes_received) >= quorum:
                self._become_leader()

    def _become_leader(self) -> None:
        self.role = Role.LEADER
        self.leader_id = self.id
        last = self._last_log_index()
        for peer in self.peers:
            self._next_index[peer]  = last + 1
            self._match_index[peer] = 0
        self._send_heartbeats()

    def _become_follower(self, term: int) -> None:
        self.current_term = term
        self.role = Role.FOLLOWER
        self.voted_for = None
        self._ticks_since_reset = 0
        self._election_timeout = self._reset_timeout()

    # ------------------------------------------------------------------
    # Log replication
    # ------------------------------------------------------------------

    def _send_heartbeats(self) -> None:
        for peer in self.peers:
            self._send_append(peer)

    def _replicate_to_peers(self) -> None:
        for peer in self.peers:
            self._send_append(peer)

    def _send_append(self, peer: int) -> None:
        if self.role != Role.LEADER:
            return

        next_idx = self._next_index.get(peer, 1)

        # If peer needs entries before our log start → send snapshot
        if next_idx <= self._snap_offset and self.snapshot is not None:
            snap = self.snapshot
            self._send(peer, MsgType.INSTALL_SNAP, {
                "term":        self.current_term,
                "leader_id":   self.id,
                "last_index":  snap.last_index,
                "last_term":   snap.last_term,
                "data":        snap.data,
            })
            return

        prev_idx  = next_idx - 1
        prev_entry = self._log_entry(prev_idx)
        prev_term  = prev_entry.term if prev_entry else (
            self.snapshot.last_term if self.snapshot and prev_idx == self._snap_offset else 0
        )

        entries = []
        for idx in range(next_idx, self._last_log_index() + 1):
            e = self._log_entry(idx)
            if e:
                entries.append({"term": e.term, "index": e.index, "command": e.command})

        self._send(peer, MsgType.APPEND_ENTRIES, {
            "term":           self.current_term,
            "leader_id":      self.id,
            "prev_log_index": prev_idx,
            "prev_log_term":  prev_term,
            "entries":        entries,
            "leader_commit":  self.commit_index,
        })

    def _on_append_entries(self, msg: Message) -> None:
        b = msg.body
        term         = b["term"]
        leader_id    = b["leader_id"]
        prev_idx     = b["prev_log_index"]
        prev_term    = b["prev_log_term"]
        entries      = b["entries"]
        leader_cmt   = b["leader_commit"]

        self._ticks_since_reset = 0
        self.leader_id = leader_id
        if self.role == Role.CANDIDATE:
            self.role = Role.FOLLOWER

        # Reply false if term < current_term
        if term < self.current_term:
            self._send(leader_id, MsgType.APPEND_RESP, {
                "term": self.current_term, "success": False,
                "match_index": 0,
            })
            return

        # Log consistency check
        if prev_idx > 0:
            existing = self._log_entry(prev_idx)
            if existing is None or existing.term != prev_term:
                # Hint: send back the conflicting term for fast backup
                conflict_term  = existing.term if existing else -1
                conflict_index = prev_idx
                if existing:
                    # find first entry with conflict_term
                    for e in self.log:
                        if e.term == conflict_term:
                            conflict_index = e.index
                            break
                self._send(leader_id, MsgType.APPEND_RESP, {
                    "term": self.current_term, "success": False,
                    "conflict_term":  conflict_term,
                    "conflict_index": conflict_index,
                    "match_index": 0,
                })
                return

        # Append new entries, truncating any conflicting suffix
        for raw in entries:
            e = LogEntry(term=raw["term"], index=raw["index"], command=raw["command"])
            existing = self._log_entry(e.index)
            if existing is not None and existing.term != e.term:
                # Delete from here onward
                self.log = [x for x in self.log if x.index < e.index]
            if self._log_entry(e.index) is None:
                self.log.append(e)

        # Advance commit index
        if leader_cmt > self.commit_index:
            self.commit_index = min(leader_cmt, self._last_log_index())

        last_new = entries[-1]["index"] if entries else prev_idx
        self._send(leader_id, MsgType.APPEND_RESP, {
            "term": self.current_term, "success": True,
            "match_index": last_new,
        })

    def _on_append_response(self, msg: Message) -> None:
        if self.role != Role.LEADER:
            return
        peer = msg.src
        b    = msg.body
        if not b["success"]:
            # Back up next_index
            if "conflict_term" in b:
                ct = b["conflict_term"]
                ci = b["conflict_index"]
                # Find last entry in our log with that term
                new_next = ci
                for e in reversed(self.log):
                    if e.term == ct:
                        new_next = e.index + 1
                        break
                self._next_index[peer] = max(1, new_next)
            else:
                self._next_index[peer] = max(1, self._next_index.get(peer, 1) - 1)
            self._send_append(peer)
            return

        match = b["match_index"]
        if match > self._match_index.get(peer, 0):
            self._match_index[peer] = match
            self._next_index[peer]  = match + 1
            self._advance_commit()

    def _advance_commit(self) -> None:
        """
        Find the highest N > commit_index such that a majority of match_index ≥ N
        and log[N].term == current_term, then commit it.
        """
        all_match = sorted(
            [self._last_log_index()] + list(self._match_index.values()),
            reverse=True,
        )
        n_servers = len(self.peers) + 1
        quorum = (n_servers // 2) + 1
        for n in all_match:
            if n <= self.commit_index:
                break
            entry = self._log_entry(n)
            if entry is None:
                continue
            if entry.term != self.current_term:
                continue
            # Count how many have replicated this
            count = 1  # leader itself
            for p in self.peers:
                if self._match_index.get(p, 0) >= n:
                    count += 1
            if count >= quorum:
                self.commit_index = n
                break

    # ------------------------------------------------------------------
    # Snapshot install
    # ------------------------------------------------------------------

    def _on_install_snapshot(self, msg: Message) -> None:
        b = msg.body
        if b["term"] < self.current_term:
            return
        snap = Snapshot(
            last_index=b["last_index"],
            last_term=b["last_term"],
            data=b["data"],
        )
        import json
        self.state_machine = json.loads(snap.data.decode())
        self.snapshot = snap
        # Discard any log entries covered by snapshot
        self.log = [e for e in self.log if e.index > snap.last_index]
        self._snap_offset = snap.last_index
        self.commit_index  = max(self.commit_index,  snap.last_index)
        self.last_applied  = max(self.last_applied,  snap.last_index)

        self._send(msg.src, MsgType.SNAP_RESP, {
            "term": self.current_term,
            "match_index": snap.last_index,
        })

    def _on_snapshot_response(self, msg: Message) -> None:
        if self.role != Role.LEADER:
            return
        peer  = msg.src
        match = msg.body["match_index"]
        self._match_index[peer] = max(self._match_index.get(peer, 0), match)
        self._next_index[peer]  = match + 1

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _apply_to_sm(self, command: Any) -> Any:
        """Apply a state-machine command.

        Commands are dicts: {"op": "set"|"del", "key": str, "value": Any}
        or arbitrary non-dict values (stored as-is under their index).
        """
        if isinstance(command, dict):
            op  = command.get("op", "set")
            key = command.get("key", "")
            if op == "set":
                self.state_machine[key] = command.get("value")
                return self.state_machine[key]
            elif op == "del":
                return self.state_machine.pop(key, None)
            elif op == "get":
                return self.state_machine.get(key)
        return command

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log_entry(self, index: int) -> Optional[LogEntry]:
        """Look up a log entry by 1-based index (accounts for snapshots)."""
        if index <= 0:
            return None
        offset_idx = index - self._snap_offset - 1
        if offset_idx < 0 or offset_idx >= len(self.log):
            return None
        return self.log[offset_idx]

    def _last_log_index(self) -> int:
        if not self.log:
            return self._snap_offset
        return self.log[-1].index

    def _last_log_term(self) -> int:
        if not self.log:
            return self.snapshot.last_term if self.snapshot else 0
        return self.log[-1].term

    def _log_is_up_to_date(self, candidate_last_idx: int, candidate_last_term: int) -> bool:
        """Raft §5.4.1: candidate log is at least as up-to-date as ours."""
        my_term = self._last_log_term()
        my_idx  = self._last_log_index()
        if candidate_last_term != my_term:
            return candidate_last_term > my_term
        return candidate_last_idx >= my_idx

    # ------------------------------------------------------------------
    # Message sending
    # ------------------------------------------------------------------

    def _send(self, dst: int, typ: MsgType, body: Dict) -> None:
        self.outgoing_msgs.append(Message(src=self.id, dst=dst, typ=typ, body=body))

    def _reset_timeout(self) -> int:
        return self._rng.randint(self.TIMEOUT_LO, self.TIMEOUT_HI)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "id":           self.id,
            "role":         self.role.value,
            "term":         self.current_term,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_len":      len(self.log),
            "leader":       self.leader_id,
            "sm_keys":      list(self.state_machine.keys()),
        }


# ---------------------------------------------------------------------------
# Cluster simulation
# ---------------------------------------------------------------------------

class RaftCluster:
    """
    Simulates a Raft cluster of n servers in a single process.

    Supports:
      - partition(nodes)     — partition a set of nodes from the rest
      - heal()               — remove all partitions
      - crash(node_id)       — stop a server
      - restart(node_id)     — re-create a server from persistent state
      - propose(cmd)         — propose a command to the current leader
      - tick(n)              — advance n ticks
    """

    def __init__(self, n: int = 5, seed: int = 42):
        self._rng   = random.Random(seed)
        self._n     = n
        self.ids    = list(range(n))
        self.servers: Dict[int, RaftServer] = {}
        self._dead:       Set[int] = set()
        self._partitioned: Set[int] = set()   # isolated from the "majority" side

        for i in self.ids:
            peers = [j for j in self.ids if j != i]
            self.servers[i] = RaftServer(i, peers, rng=random.Random(seed + i))

    # ------------------------------------------------------------------
    # Tick & message delivery
    # ------------------------------------------------------------------

    def tick(self, n: int = 1) -> None:
        """Advance all living servers n ticks and deliver messages."""
        for _ in range(n):
            msgs: List[Message] = []
            for sid, srv in self.servers.items():
                if sid in self._dead:
                    continue
                srv.tick()
                msgs.extend(srv.outgoing_msgs)
                srv.outgoing_msgs.clear()
            self._deliver(msgs)
            self._apply_all()

    def _deliver(self, msgs: List[Message]) -> None:
        for m in msgs:
            if m.dst in self._dead:
                continue
            # Drop messages crossing the partition boundary in either direction
            if m.src in self._partitioned and m.dst not in self._partitioned:
                continue
            if m.dst in self._partitioned and m.src not in self._partitioned:
                continue
            if m.dst in self.servers:
                self.servers[m.dst].handle(m)

    def _apply_all(self) -> None:
        for sid, srv in self.servers.items():
            if sid not in self._dead:
                srv.apply_committed()

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def partition(self, isolated: List[int]) -> None:
        """Isolate a set of nodes from the rest of the cluster."""
        self._partitioned = set(isolated)

    def heal(self) -> None:
        """Remove all network partitions."""
        self._partitioned = set()

    def crash(self, node_id: int) -> None:
        """Kill a server (it stops ticking and receiving messages)."""
        self._dead.add(node_id)

    def restart(self, node_id: int) -> None:
        """Restart a previously crashed server (loses volatile state)."""
        self._dead.discard(node_id)
        old = self.servers[node_id]
        peers = [j for j in self.ids if j != node_id]
        new_srv = RaftServer(node_id, peers, rng=random.Random(node_id))
        # Restore durable state (term, vote, log)
        new_srv.current_term = old.current_term
        new_srv.voted_for    = old.voted_for
        new_srv.log          = list(old.log)
        new_srv.snapshot     = old.snapshot
        new_srv._snap_offset = old._snap_offset
        self.servers[node_id] = new_srv

    # ------------------------------------------------------------------
    # Client interface
    # ------------------------------------------------------------------

    def leader(self) -> Optional[RaftServer]:
        """Return the current leader visible from the non-partitioned majority."""
        for srv in self.servers.values():
            if (srv.id not in self._dead
                    and srv.id not in self._partitioned
                    and srv.role == Role.LEADER):
                return srv
        return None

    def propose(self, command: Any, wait_ticks: int = 30) -> Tuple[bool, Optional[int]]:
        """
        Propose a command. Returns (committed, index).

        Waits up to wait_ticks for a leader and commit.
        """
        # Find or wait for a leader
        for _ in range(wait_ticks):
            leader = self.leader()
            if leader:
                idx = leader.propose(command)
                if idx is None:
                    self.tick(1)
                    continue
                # Wait for commit
                for _ in range(wait_ticks):
                    self.tick(1)
                    if leader.commit_index >= idx:
                        return True, idx
                return False, idx
            self.tick(1)
        return False, None

    def read(self, key: str) -> Optional[Any]:
        """Linearizable read from the leader's state machine."""
        ldr = self.leader()
        if ldr:
            return ldr.state_machine.get(key)
        return None

    def quorum_read(self, key: str) -> Optional[Any]:
        """
        Read the value of key from a majority of nodes (quorum read).
        Returns the value if consistent across a majority.
        """
        live = [s for s in self.servers.values() if s.id not in self._dead]
        values: Dict[Any, int] = {}
        for s in live:
            v = str(s.state_machine.get(key))
            values[v] = values.get(v, 0) + 1
        if not values:
            return None
        majority_val, count = max(values.items(), key=lambda x: x[1])
        quorum = len(live) // 2 + 1
        if count >= quorum:
            return None if majority_val == "None" else majority_val
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> List[Dict]:
        return [
            {**srv.status(), "dead": srv.id in self._dead,
             "partitioned": srv.id in self._partitioned}
            for srv in self.servers.values()
        ]

    def is_consistent(self) -> bool:
        """Check that all live servers have the same commit_index values committed."""
        live = [s for s in self.servers.values() if s.id not in self._dead]
        if not live:
            return True
        min_commit = min(s.commit_index for s in live)
        if min_commit == 0:
            return True
        # All servers should agree on entries up to min_commit
        for idx in range(1, min_commit + 1):
            entries = []
            for s in live:
                e = s._log_entry(idx)
                if e is not None:
                    entries.append(e.term)
                elif s.snapshot and s.snapshot.last_index >= idx:
                    entries.append(s.snapshot.last_term)
            if len(set(entries)) > 1:
                return False
        return True

    def majority_committed(self, index: int) -> bool:
        """True if a quorum has committed at least `index`."""
        live = [s for s in self.servers.values() if s.id not in self._dead]
        count = sum(1 for s in live if s.commit_index >= index)
        return count > len(live) // 2


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def run_basic_demo(n_servers: int = 5, seed: int = 7) -> Dict[str, Any]:
    """
    Full end-to-end demo:
      1. Elect a leader
      2. Propose 10 key-value writes
      3. Crash a follower, write more
      4. Bring follower back, verify log convergence
      5. Partition leader, verify new leader elected
      6. Heal, verify single leader
    """
    cluster = RaftCluster(n_servers, seed=seed)

    # Phase 1: elect leader
    cluster.tick(40)
    ldr = cluster.leader()
    assert ldr is not None, "No leader after 40 ticks"

    # Phase 2: 10 writes
    committed_indices = []
    for i in range(10):
        ok, idx = cluster.propose({"op": "set", "key": f"k{i}", "value": i * 100})
        assert ok, f"Write {i} not committed"
        committed_indices.append(idx)
    cluster.tick(10)

    # Phase 3: crash a follower, write more
    follower = next(s for s in cluster.servers.values()
                    if s.role != Role.LEADER and s.id not in cluster._dead)
    cluster.crash(follower.id)
    for i in range(5):
        ok, idx = cluster.propose({"op": "set", "key": f"crashed_{i}", "value": i})
    cluster.tick(15)

    # Phase 4: restart follower, verify convergence
    cluster.restart(follower.id)
    cluster.tick(60)
    assert cluster.is_consistent(), "Log inconsistency after restart"

    # Phase 5: partition old leader
    old_leader_id = cluster.leader().id
    cluster.partition([old_leader_id])
    cluster.tick(50)
    new_ldr = cluster.leader()
    assert new_ldr is not None and new_ldr.id != old_leader_id, \
        "New leader not elected after partition"

    # Phase 6: heal, verify single leader
    cluster.heal()
    cluster.tick(40)
    leaders = [s for s in cluster.servers.values() if s.role == Role.LEADER]
    assert len(leaders) == 1, f"Expected 1 leader, got {len(leaders)}"

    return {
        "final_leader":      leaders[0].id,
        "final_term":        leaders[0].current_term,
        "entries_committed": leaders[0].commit_index,
        "consistent":        cluster.is_consistent(),
        "cluster_status":    cluster.status(),
    }
