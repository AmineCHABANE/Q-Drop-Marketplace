"""
Q-STREAM: Windowed Stream Processor — Reference Implementation

Stream processing is the engine behind Apache Kafka Streams, Apache Flink, and
Google Dataflow. The core problem: process an unbounded stream of events with
stateful computations over time windows.

This implementation demonstrates:
- Tumbling windows (fixed-size, non-overlapping)
- Sliding windows (fixed-size, overlapping)
- Session windows (gap-based, variable size)
- Watermarks (handling late/out-of-order events)
- Stateful operators (running counts, running averages)
- The Dataflow programming model (transform → window → trigger → aggregate)

The "Dataflow model" (Akidau et al., 2015) is the theoretical foundation of
both Apache Beam and Google Cloud Dataflow. Understanding it unlocks all
modern stream processing systems.

Usage:
    from q_stream import StreamProcessor, Event

    def handle_window(window_result):
        print(f"Window [{window_result.start}, {window_result.end}]: {window_result.value}")

    proc = StreamProcessor(window_size_ms=5000, slide_ms=1000)
    proc.on_window(handle_window)

    for timestamp, value in events:
        proc.emit(Event(timestamp=timestamp, key="sensor_1", value=value))

    proc.flush()  # trigger final windows
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """
    A single stream event.

    timestamp : event time in milliseconds (when the event actually happened,
                NOT when it was processed — this distinction is crucial)
    key       : partition key (e.g. user_id, sensor_id)
    value     : payload (any Python value)
    """
    timestamp: int
    key: str
    value: Any


@dataclass
class WindowResult:
    """Output of a window computation."""
    key: str
    start: int      # window start timestamp (inclusive), ms
    end: int        # window end timestamp (exclusive), ms
    count: int
    value: Any      # aggregated value


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------

class Watermark:
    """
    Tracks event-time progress to handle late data.

    The watermark W(t) means: "no event with timestamp < W(t) will arrive."
    When the watermark passes a window's end time, that window is closed.

    In practice, watermarks are heuristic (we can't know the future).
    A common strategy: watermark = max_seen_event_time - allowed_lateness.

    Real systems (Flink, Dataflow) support custom WatermarkGenerators.
    Here we use a simple max-minus-lateness strategy.
    """

    def __init__(self, allowed_lateness_ms: int = 0) -> None:
        self.allowed_lateness_ms = allowed_lateness_ms
        self._max_event_time: int = 0

    def update(self, event_time: int) -> None:
        if event_time > self._max_event_time:
            self._max_event_time = event_time

    @property
    def current(self) -> int:
        """Current watermark value."""
        return self._max_event_time - self.allowed_lateness_ms


# ---------------------------------------------------------------------------
# Window assigners
# ---------------------------------------------------------------------------

def tumbling_windows(event_time: int, size_ms: int) -> List[Tuple[int, int]]:
    """
    Assign event to exactly one tumbling window.

    Tumbling: [0,5), [5,10), [10,15), ...
    Each event belongs to exactly one window. Windows don't overlap.
    Use for: throughput counting, fixed-period aggregations.
    """
    window_start = (event_time // size_ms) * size_ms
    return [(window_start, window_start + size_ms)]


def sliding_windows(
    event_time: int, size_ms: int, slide_ms: int
) -> List[Tuple[int, int]]:
    """
    Assign event to multiple overlapping sliding windows.

    Sliding (size=10s, slide=5s): [0,10), [5,15), [10,20), ...
    Each event appears in size/slide windows. Overlapping.
    Use for: moving averages, rolling statistics.
    """
    # Earliest window that contains this event
    first_start = ((event_time - size_ms) // slide_ms + 1) * slide_ms
    windows = []
    start = first_start
    while start <= event_time:
        if start + size_ms > event_time:
            windows.append((start, start + size_ms))
        start += slide_ms
    return windows


# ---------------------------------------------------------------------------
# State store (per-window accumulator)
# ---------------------------------------------------------------------------

@dataclass
class WindowState:
    """Mutable accumulator for one (key, window) combination."""
    key: str
    start: int
    end: int
    count: int = 0
    total: Any = 0
    min_val: Any = None
    max_val: Any = None
    _values: List[Any] = field(default_factory=list)

    def add(self, value: Any) -> None:
        self.count += 1
        self._values.append(value)
        try:
            self.total += value
            if self.min_val is None or value < self.min_val:
                self.min_val = value
            if self.max_val is None or value > self.max_val:
                self.max_val = value
        except TypeError:
            pass  # non-numeric value

    @property
    def avg(self) -> Optional[float]:
        return self.total / self.count if self.count > 0 else None

    def to_result(self, agg: str = "sum") -> WindowResult:
        value = {
            "sum": self.total,
            "count": self.count,
            "avg": self.avg,
            "min": self.min_val,
            "max": self.max_val,
            "values": list(self._values),
        }.get(agg, self.total)
        return WindowResult(
            key=self.key, start=self.start, end=self.end,
            count=self.count, value=value
        )


# ---------------------------------------------------------------------------
# Stream Processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """
    Windowed stream processor supporting tumbling and sliding windows.

    Architecture:
        emit(event)
          → assign to windows
          → update state per (key, window)
          → advance watermark
          → close and emit windows where watermark > window.end

    State is kept in a dict of WindowState objects, keyed by (key, start, end).
    In Flink this would be backed by RocksDB for fault tolerance.

    Example — count events per user per 5-second tumbling window:

        proc = StreamProcessor(window_type="tumbling", window_size_ms=5000)
        proc.on_window(lambda r: print(r))
        for ts, uid, val in stream:
            proc.emit(Event(ts, uid, val))
        proc.flush()
    """

    def __init__(
        self,
        window_type: str = "tumbling",
        window_size_ms: int = 5000,
        slide_ms: Optional[int] = None,
        allowed_lateness_ms: int = 0,
        agg: str = "sum",
    ) -> None:
        """
        window_type         : 'tumbling' | 'sliding'
        window_size_ms      : window duration
        slide_ms            : slide interval (sliding windows only; default = size/2)
        allowed_lateness_ms : how late an event can arrive and still be counted
        agg                 : aggregation function for window output
                              'sum' | 'count' | 'avg' | 'min' | 'max' | 'values'
        """
        self.window_type = window_type
        self.window_size_ms = window_size_ms
        self.slide_ms = slide_ms or window_size_ms // 2
        self.agg = agg

        self._watermark = Watermark(allowed_lateness_ms)
        self._state: Dict[Tuple[str, int, int], WindowState] = {}
        self._callbacks: List[Callable[[WindowResult], None]] = []
        self._dropped_late: int = 0

    def on_window(self, callback: Callable[[WindowResult], None]) -> None:
        """Register a callback called when a window closes."""
        self._callbacks.append(callback)

    def emit(self, event: Event) -> None:
        """
        Process one event.

        1. Advance watermark
        2. Check if event is too late (drop it if so)
        3. Assign to windows
        4. Update each window's state
        5. Close windows whose end ≤ new watermark
        """
        self._watermark.update(event.timestamp)
        wm = self._watermark.current

        # Drop events that arrived after their windows already closed
        if event.timestamp < wm:
            self._dropped_late += 1
            return

        # Assign to windows
        if self.window_type == "tumbling":
            windows = tumbling_windows(event.timestamp, self.window_size_ms)
        elif self.window_type == "sliding":
            windows = sliding_windows(event.timestamp, self.window_size_ms, self.slide_ms)
        else:
            raise ValueError(f"Unknown window_type '{self.window_type}'")

        for w_start, w_end in windows:
            state_key = (event.key, w_start, w_end)
            if state_key not in self._state:
                self._state[state_key] = WindowState(
                    key=event.key, start=w_start, end=w_end
                )
            self._state[state_key].add(event.value)

        # Fire and evict windows whose end time has passed the watermark
        self._fire_ready_windows(wm)

    def _fire_ready_windows(self, watermark: int) -> None:
        """Close all windows where window.end <= watermark."""
        ready = [k for k, s in self._state.items() if s.end <= watermark]
        for key in ready:
            state = self._state.pop(key)
            result = state.to_result(self.agg)
            for cb in self._callbacks:
                cb(result)

    def flush(self) -> None:
        """
        Force-close all remaining open windows.
        Call at end-of-stream to drain any windows that didn't fire.
        """
        for state in self._state.values():
            result = state.to_result(self.agg)
            for cb in self._callbacks:
                cb(result)
        self._state.clear()

    def stats(self) -> dict:
        return {
            "open_windows": len(self._state),
            "watermark_ms": self._watermark.current,
            "max_event_time_ms": self._watermark._max_event_time,
            "dropped_late_events": self._dropped_late,
        }


# ---------------------------------------------------------------------------
# Session Window Processor (bonus: gap-based windows)
# ---------------------------------------------------------------------------

class SessionWindowProcessor:
    """
    Session windows: group events by inactivity gaps rather than fixed time.

    A new session starts when the gap since the last event for a key exceeds
    the session gap. Used for: user session analytics, IoT burst detection.

    Example (gap=5s): events at t=1,2,3 → session [1,3], then t=10,11 → session [10,11]
    """

    def __init__(self, gap_ms: int = 5000) -> None:
        self.gap_ms = gap_ms
        self._sessions: Dict[str, WindowState] = {}
        self._callbacks: List[Callable[[WindowResult], None]] = []

    def on_window(self, callback: Callable[[WindowResult], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, event: Event) -> None:
        key = event.key
        if key in self._sessions:
            session = self._sessions[key]
            gap = event.timestamp - session.end
            if gap > self.gap_ms:
                # Gap exceeded: close old session, start new one
                result = session.to_result()
                for cb in self._callbacks:
                    cb(result)
                self._sessions[key] = WindowState(
                    key=key, start=event.timestamp, end=event.timestamp
                )
            else:
                # Extend session
                session.end = event.timestamp
        else:
            self._sessions[key] = WindowState(
                key=key, start=event.timestamp, end=event.timestamp
            )
        self._sessions[key].add(event.value)

    def flush(self) -> None:
        for session in self._sessions.values():
            result = session.to_result()
            for cb in self._callbacks:
                cb(result)
        self._sessions.clear()
