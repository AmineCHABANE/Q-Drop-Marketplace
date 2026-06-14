"""
Q-REGEX — A regular-expression engine via Thompson NFA construction
Ken Thompson, "Regular Expression Search Algorithm", CACM 1968
(popularized by Russ Cox, "Regular Expression Matching Can Be Simple And Fast")

Compiles a regex to a Nondeterministic Finite Automaton and simulates it by
advancing a *set* of states in lockstep over the input. This guarantees linear
O(n·m) time with NO catastrophic backtracking — the pathological blow-up that
takes down production systems using backtracking engines (the 2016 Stack
Overflow and 2019 Cloudflare outages were both regex backtracking).

Supported syntax:
  literals   .  *  +  ?  |  ( )   character classes [a-z0-9]  [^abc]
  escapes \\d \\w \\s \\. \\* (and \\ before any metachar)
  anchors  ^  $

This is the algorithmic heart of grep, RE2, and lexer generators.
Zero dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

class _Re:                       # base
    pass

@dataclass
class _Char(_Re):
    matcher: "callable"          # (ch) -> bool
    label: str

@dataclass
class _Concat(_Re):
    parts: List[_Re]

@dataclass
class _Alt(_Re):
    options: List[_Re]

@dataclass
class _Star(_Re):
    node: _Re
    greedy_plus: bool = False    # + is Star with a leading copy
    optional: bool = False       # ? marker

@dataclass
class _Anchor(_Re):
    at: str                      # "^" or "$"


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, pattern: str):
        self.s = pattern
        self.i = 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def next(self) -> str:
        ch = self.s[self.i]
        self.i += 1
        return ch

    def parse(self) -> _Re:
        node = self._alt()
        if self.i != len(self.s):
            raise ValueError(f"unexpected char at {self.i}: {self.peek()!r}")
        return node

    def _alt(self) -> _Re:
        options = [self._concat()]
        while self.peek() == "|":
            self.next()
            options.append(self._concat())
        return options[0] if len(options) == 1 else _Alt(options)

    def _concat(self) -> _Re:
        parts: List[_Re] = []
        while self.peek() and self.peek() not in "|)":
            parts.append(self._repeat())
        if len(parts) == 1:
            return parts[0]
        return _Concat(parts)

    def _repeat(self) -> _Re:
        node = self._atom()
        # NB: tuple membership, not `in "*+?"` — "" is a substring of every str
        while self.peek() in ("*", "+", "?"):
            q = self.next()
            if q == "*":
                node = _Star(node)
            elif q == "+":
                node = _Star(node, greedy_plus=True)
            else:  # ?
                node = _Star(node, optional=True)
        return node

    def _atom(self) -> _Re:
        ch = self.peek()
        if ch == "(":
            self.next()
            node = self._alt()
            if self.peek() != ")":
                raise ValueError("unbalanced '('")
            self.next()
            return node
        if ch == "[":
            return self._char_class()
        if ch in ("^", "$"):
            self.next()
            return _Anchor(ch)
        if ch == ".":
            self.next()
            return _Char(lambda c: c != "\n", ".")
        if ch == "\\":
            self.next()
            return self._escape(self.next())
        if ch in (")", "|"):
            raise ValueError(f"unexpected {ch!r}")
        if ch == "":
            raise ValueError("unexpected end of pattern")
        self.next()
        return _Char(lambda c, lit=ch: c == lit, ch)

    def _escape(self, e: str) -> _Char:
        classes = {
            "d": (lambda c: c.isdigit(), "\\d"),
            "w": (lambda c: c.isalnum() or c == "_", "\\w"),
            "s": (lambda c: c.isspace(), "\\s"),
            "D": (lambda c: not c.isdigit(), "\\D"),
            "W": (lambda c: not (c.isalnum() or c == "_"), "\\W"),
            "S": (lambda c: not c.isspace(), "\\S"),
        }
        if e in classes:
            fn, lab = classes[e]
            return _Char(fn, lab)
        return _Char(lambda c, lit=e: c == lit, e)   # escaped literal

    def _char_class(self) -> _Char:
        self.next()                       # consume '['
        negate = False
        if self.peek() == "^":
            negate = True
            self.next()
        ranges = []
        singles = set()
        class_fns = []
        while self.peek() and self.peek() != "]":
            c = self.next()
            if c == "\\":
                esc = self._escape(self.next())
                class_fns.append(esc.matcher)
                continue
            if self.peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.next()               # consume '-'
                hi = self.next()
                ranges.append((c, hi))
            else:
                singles.add(c)
        if self.peek() != "]":
            raise ValueError("unbalanced '['")
        self.next()                       # consume ']'

        def matches(ch: str) -> bool:
            r = (ch in singles
                 or any(lo <= ch <= hi for lo, hi in ranges)
                 or any(fn(ch) for fn in class_fns))
            return (not r) if negate else r

        label = f"[{'^' if negate else ''}…]"
        return _Char(matches, label)


# ---------------------------------------------------------------------------
# NFA (Thompson construction)
# ---------------------------------------------------------------------------

@dataclass
class _State:
    matcher: Optional["callable"] = None     # char predicate, or None for split/match
    out: Optional["_State"] = None
    out2: Optional["_State"] = None
    anchor: Optional[str] = None
    is_match: bool = False


@dataclass
class _Frag:
    start: "_State"
    outs: List[list]                         # list of [state, attr] dangling pointers


def _patch(outs, target) -> None:
    for st, attr in outs:
        setattr(st, attr, target)


class _Compiler:
    def build(self, node: _Re) -> _Frag:
        if isinstance(node, _Char):
            s = _State(matcher=node.matcher)
            return _Frag(s, [[s, "out"]])
        if isinstance(node, _Anchor):
            s = _State(anchor=node.at)
            return _Frag(s, [[s, "out"]])
        if isinstance(node, _Concat):
            frags = [self.build(p) for p in node.parts]
            for a, b in zip(frags, frags[1:]):
                _patch(a.outs, b.start)
            return _Frag(frags[0].start, frags[-1].outs)
        if isinstance(node, _Alt):
            return self._alt_binary(node.options)
        if isinstance(node, _Star):
            return self._build_quant(node)
        raise ValueError(f"unknown node {node}")

    def _alt_binary(self, options: List[_Re]) -> _Frag:
        frag = self.build(options[0])
        for opt in options[1:]:
            fr = self.build(opt)
            split = _State(out=frag.start, out2=fr.start)
            frag = _Frag(split, frag.outs + fr.outs)
        return frag

    def _build_quant(self, node: _Star) -> _Frag:
        inner = self.build(node.node)
        if node.optional:                     # ?  — zero or one
            split = _State(out=inner.start)
            return _Frag(split, inner.outs + [[split, "out2"]])
        if node.greedy_plus:                  # +  — one or more
            split = _State(out=inner.start)
            _patch(inner.outs, split)
            return _Frag(inner.start, [[split, "out2"]])
        # *  — zero or more
        split = _State(out=inner.start)
        _patch(inner.outs, split)
        return _Frag(split, [[split, "out2"]])


def _compile(pattern: str) -> _State:
    ast = _Parser(pattern).parse()
    frag = _Compiler().build(ast)
    match_state = _State(is_match=True)
    _patch(frag.outs, match_state)
    return frag.start


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _add_state(state: Optional[_State], states: list, seen: set,
               pos: int, n: int) -> None:
    """Epsilon-closure: follow splits and satisfied anchors."""
    if state is None or id(state) in seen:
        return
    seen.add(id(state))
    if state.matcher is None and not state.is_match and state.anchor is None:
        # split state
        _add_state(state.out, states, seen, pos, n)
        _add_state(state.out2, states, seen, pos, n)
        return
    if state.anchor is not None:
        ok = (state.anchor == "^" and pos == 0) or (state.anchor == "$" and pos == n)
        if ok:
            _add_state(state.out, states, seen, pos, n)
        return
    states.append(state)


class Regex:
    """A compiled regular expression."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._start = _compile(pattern)

    def fullmatch(self, text: str) -> bool:
        """True iff the entire `text` matches the pattern."""
        n = len(text)
        clos: list = []
        _add_state(self._start, clos, set(), 0, n)
        current = clos
        for pos, ch in enumerate(text):
            nxt: list = []
            seen: set = set()
            for st in current:
                if st.matcher and st.matcher(ch):
                    _add_state(st.out, nxt, seen, pos + 1, n)
            current = nxt
            if not current:
                break
        return any(st.is_match for st in current)

    def search(self, text: str) -> bool:
        """True iff the pattern matches anywhere in `text` (substring)."""
        # Anchored patterns are handled by ^/$ during simulation; for a plain
        # search we try every starting offset (still linear per offset).
        n = len(text)
        for start in range(n + 1):
            clos: list = []
            _add_state(self._start, clos, set(), start, n)
            current = clos
            if any(st.is_match for st in current):
                return True
            for pos in range(start, n):
                ch = text[pos]
                nxt: list = []
                seen: set = set()
                for st in current:
                    if st.matcher and st.matcher(ch):
                        _add_state(st.out, nxt, seen, pos + 1, n)
                current = nxt
                if not current:
                    break
                if any(st.is_match for st in current):
                    return True
        return False

    def __repr__(self) -> str:
        return f"Regex({self.pattern!r})"


def fullmatch(pattern: str, text: str) -> bool:
    return Regex(pattern).fullmatch(text)


def search(pattern: str, text: str) -> bool:
    return Regex(pattern).search(text)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_regex() -> dict:
    """
    Compile a few patterns and match them — and show the headline property:
    a pattern that makes backtracking engines explode, (a+)+$, matches a long
    non-matching input in linear time here, with zero blow-up.
    """
    email = Regex(r"\w+@\w+\.\w+")
    ipish = Regex(r"\d+\.\d+\.\d+\.\d+")
    alt = Regex(r"(cat|dog|bird)s?")

    # The classic catastrophic-backtracking pattern, on input that does NOT match.
    # A backtracking engine takes exponential time; an NFA simulation is linear.
    evil = Regex(r"(a+)+$")
    evil_input = "a" * 40 + "!"      # would hang a backtracking engine

    return {
        "email_match": email.fullmatch("user@example.com"),
        "email_reject": not email.fullmatch("not-an-email"),
        "ip_search": ipish.search("connect to 10.0.0.1 now"),
        "alt_match": alt.fullmatch("dogs") and alt.fullmatch("cat"),
        "alt_reject": not alt.fullmatch("fish"),
        "anchored_class": Regex(r"^[A-Z][a-z]+$").fullmatch("Quantum"),
        "no_catastrophic_backtracking": evil.fullmatch(evil_input) is False,
    }
