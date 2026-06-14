"""
Q-RATELIMIT — Rate limiting algorithms (the gatekeeper of every API)
The four classic algorithms every API gateway, load balancer and database
connection pool implements to protect a service from overload and abuse:

  • TokenBucket          — allows controlled bursts, smooth average (Stripe, AWS)
  • LeakyBucket          — perfectly smooth output, queues bursts (traffic shaping)
  • FixedWindowCounter   — simplest; suffers boundary bursts
  • SlidingWindowLog     — exact, memory-heavy (stores each hit timestamp)
  • SlidingWindowCounter — approximate, O(1) memory (Cloudflare's choice)

The clock is injectable, so behaviour is fully deterministic and testable —
no sleeping in tests. In production you pass time.monotonic.

Zero dependencies.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional


Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    A bucket holds up to `capacity` tokens and refills at `rate` tokens/second.
    Each request costs tokens; if enough are available it is allowed, else denied.

    Permits short bursts up to `capacity` while bounding the long-run average to
    `rate`. This is what AWS API Gateway, Stripe and most cloud APIs use.
    """

    def __init__(self, rate: float, capacity: float,
                 clock: Optional[Clock] = None):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._clock = clock or time.monotonic
        self._last = self._clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    def allow(self, cost: float = 1.0) -> bool:
        """Try to consume `cost` tokens. Returns True if allowed."""
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def __repr__(self) -> str:
        return f"TokenBucket(rate={self.rate}/s, capacity={self.capacity})"


# ---------------------------------------------------------------------------
# Leaky bucket (as a queue / meter)
# ---------------------------------------------------------------------------

class LeakyBucket:
    """
    Requests fill a bucket that leaks at a constant `rate` per second. If adding
    a request would overflow `capacity`, it is rejected. Output is perfectly
    smooth regardless of input burstiness — classic traffic shaping.
    """

    def __init__(self, rate: float, capacity: float,
                 clock: Optional[Clock] = None):
        self.rate = rate
        self.capacity = capacity
        self._level = 0.0
        self._clock = clock or time.monotonic
        self._last = self._clock()

    def _leak(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._level = max(0.0, self._level - elapsed * self.rate)
            self._last = now

    def allow(self, amount: float = 1.0) -> bool:
        self._leak()
        if self._level + amount <= self.capacity:
            self._level += amount
            return True
        return False

    @property
    def level(self) -> float:
        self._leak()
        return self._level

    def __repr__(self) -> str:
        return f"LeakyBucket(rate={self.rate}/s, capacity={self.capacity})"


# ---------------------------------------------------------------------------
# Fixed window counter
# ---------------------------------------------------------------------------

class FixedWindowCounter:
    """
    Count requests within fixed clock windows of `window` seconds; allow up to
    `limit` per window. Simplest and cheapest, but permits up to 2× `limit`
    across a window boundary (the burst that motivates sliding windows).
    """

    def __init__(self, limit: int, window: float,
                 clock: Optional[Clock] = None):
        self.limit = limit
        self.window = window
        self._clock = clock or time.monotonic
        self._win_start = self._clock()
        self._count = 0

    def allow(self) -> bool:
        now = self._clock()
        if now - self._win_start >= self.window:
            self._win_start = now
            self._count = 0
        if self._count < self.limit:
            self._count += 1
            return True
        return False


# ---------------------------------------------------------------------------
# Sliding window log (exact)
# ---------------------------------------------------------------------------

class SlidingWindowLog:
    """
    Keep a timestamp for every allowed request; a new request is allowed iff
    fewer than `limit` timestamps fall within the trailing `window` seconds.
    Exact, but memory grows with throughput.
    """

    def __init__(self, limit: int, window: float,
                 clock: Optional[Clock] = None):
        self.limit = limit
        self.window = window
        self._clock = clock or time.monotonic
        self._hits: Deque[float] = deque()

    def allow(self) -> bool:
        now = self._clock()
        cutoff = now - self.window
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()
        if len(self._hits) < self.limit:
            self._hits.append(now)
            return True
        return False

    @property
    def current(self) -> int:
        return len(self._hits)


# ---------------------------------------------------------------------------
# Sliding window counter (approximate, O(1) memory)
# ---------------------------------------------------------------------------

class SlidingWindowCounter:
    """
    Cloudflare's approach: blend the current and previous fixed-window counts,
    weighting the previous window by how much of it still overlaps the trailing
    window. O(1) memory, smooths boundary bursts, accurate within a few percent.
    """

    def __init__(self, limit: int, window: float,
                 clock: Optional[Clock] = None):
        self.limit = limit
        self.window = window
        self._clock = clock or time.monotonic
        self._cur_start = self._clock()
        self._cur = 0
        self._prev = 0

    def _roll(self, now: float) -> None:
        elapsed = now - self._cur_start
        if elapsed >= self.window:
            if elapsed >= 2 * self.window:
                self._prev = 0
            else:
                self._prev = self._cur
            self._cur = 0
            self._cur_start = now - (elapsed % self.window)

    def allow(self) -> bool:
        now = self._clock()
        self._roll(now)
        # Fraction of the previous window still inside the sliding window
        elapsed_in_cur = now - self._cur_start
        prev_weight = max(0.0, (self.window - elapsed_in_cur) / self.window)
        estimated = self._prev * prev_weight + self._cur
        if estimated < self.limit:
            self._cur += 1
            return True
        return False


# ---------------------------------------------------------------------------
# A manual clock for deterministic testing / demos
# ---------------------------------------------------------------------------

class ManualClock:
    """A controllable clock: advance(dt) instead of waiting on wall time."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_rate_limit() -> dict:
    """
    Drive a token bucket with a manual clock: burst through the initial
    capacity, get throttled, wait for a refill, then succeed again — exactly
    the behaviour an API client experiences against a rate-limited endpoint.
    """
    clock = ManualClock()
    # 5 tokens/sec, burst capacity 10
    tb = TokenBucket(rate=5, capacity=10, clock=clock)

    # Instant burst of 12 requests at t=0: first 10 allowed, last 2 denied
    burst = [tb.allow() for _ in range(12)]
    allowed_burst = sum(burst)

    # After 1 second, ~5 tokens have refilled
    clock.advance(1.0)
    refilled = [tb.allow() for _ in range(6)]
    allowed_after_refill = sum(refilled)

    # Sliding-window log: 3 per second, exact
    clock2 = ManualClock()
    swl = SlidingWindowLog(limit=3, window=1.0, clock=clock2)
    first3 = [swl.allow() for _ in range(3)]      # all allowed
    fourth = swl.allow()                          # denied
    clock2.advance(1.01)                          # window slides past the first 3
    fifth = swl.allow()                           # allowed again

    return {
        "burst_allowed": allowed_burst,          # expect 10 (== capacity)
        "burst_denied": 12 - allowed_burst,      # expect 2
        "allowed_after_1s_refill": allowed_after_refill,  # expect ~5
        "sliding_log_first3_allowed": all(first3),
        "sliding_log_fourth_denied": not fourth,
        "sliding_log_recovers_after_window": fifth,
    }
