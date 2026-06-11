"""
Q-COL: Columnar Data Engine — Reference Implementation

Columnar storage is the foundation of DuckDB, Apache Arrow, Parquet, and Polars.
The key insight: storing data column-by-column (not row-by-row) allows analytical
queries to read only the columns they need and apply SIMD-friendly vectorized operations.

This implementation demonstrates:
- Column-oriented storage using Python lists
- Vectorized filter evaluation (predicate pushdown)
- Vectorized aggregation (sum, avg, min, max, count)
- GROUP BY via hash aggregation
- Dictionary encoding for string columns (major compression win)
- Run-length encoding for sparse/repeated data

No external dependencies. Pure Python for clarity.

Usage:
    from q_col import ColumnStore

    store = ColumnStore()
    store.add_column("city", ["Paris", "London", "Paris", "Berlin", "London"])
    store.add_column("sales", [100, 200, 150, 300, 250])
    store.add_column("qty",   [1, 2, 1, 3, 2])

    # Filter: city == "Paris"
    mask = store.filter("city", "==", "Paris")
    print(store.select(["city", "sales"], mask))

    # Aggregation
    print(store.aggregate("sales", "sum", mask))   # 250

    # Group by
    result = store.groupby("city", {"sales": "sum", "qty": "avg"})
    for row in result:
        print(row)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bitmap mask (row selection)
# ---------------------------------------------------------------------------

class Mask:
    """
    A boolean array over rows. Used to pass selections between operations
    without materializing intermediate result sets.

    In Arrow this is a validity bitmap (packed bits). Here we use a Python
    list of bools — same semantics, easier to read.
    """

    def __init__(self, bits: List[bool]) -> None:
        self.bits = bits

    def __and__(self, other: "Mask") -> "Mask":
        return Mask([a and b for a, b in zip(self.bits, other.bits)])

    def __or__(self, other: "Mask") -> "Mask":
        return Mask([a or b for a, b in zip(self.bits, other.bits)])

    def __invert__(self) -> "Mask":
        return Mask([not b for b in self.bits])

    def count(self) -> int:
        return sum(self.bits)

    def indices(self) -> List[int]:
        return [i for i, b in enumerate(self.bits) if b]

    @classmethod
    def all_true(cls, n: int) -> "Mask":
        return cls([True] * n)

    @classmethod
    def all_false(cls, n: int) -> "Mask":
        return cls([False] * n)


# ---------------------------------------------------------------------------
# Dictionary-encoded column
# ---------------------------------------------------------------------------

class DictColumn:
    """
    Dictionary encoding for string/categorical columns.

    Instead of storing "Paris", "Paris", "London", "Paris" (4 strings),
    we store:
        dict = {0: "Paris", 1: "London"}
        codes = [0, 0, 1, 0]

    For high-cardinality columns this can achieve 10–100x compression.
    Equality filters and group-by operate on integer codes, not strings.
    """

    def __init__(self, values: List[Any]) -> None:
        self._dict: Dict[int, Any] = {}       # code → value
        self._reverse: Dict[Any, int] = {}    # value → code
        self._codes: List[int] = []

        for v in values:
            if v not in self._reverse:
                code = len(self._dict)
                self._dict[code] = v
                self._reverse[v] = code
            self._codes.append(self._reverse[v])

    def get(self, idx: int) -> Any:
        return self._dict[self._codes[idx]]

    def filter_eq(self, value: Any) -> "Mask":
        """Vectorized equality filter: compare codes, not strings."""
        if value not in self._reverse:
            return Mask.all_false(len(self._codes))
        code = self._reverse[value]
        return Mask([c == code for c in self._codes])

    def filter_ne(self, value: Any) -> "Mask":
        if value not in self._reverse:
            return Mask.all_true(len(self._codes))
        code = self._reverse[value]
        return Mask([c != code for c in self._codes])

    def to_list(self, mask: Optional[Mask] = None) -> List[Any]:
        if mask is None:
            return [self._dict[c] for c in self._codes]
        return [self._dict[self._codes[i]] for i in mask.indices()]

    def __len__(self) -> int:
        return len(self._codes)

    def cardinality(self) -> int:
        return len(self._dict)


# ---------------------------------------------------------------------------
# Run-length encoded column
# ---------------------------------------------------------------------------

class RLEColumn:
    """
    Run-length encoding: represent [1, 1, 1, 2, 2, 3] as [(1,3), (2,2), (3,1)].

    Efficient for sorted or repetitive numeric columns (timestamps, IDs after sort).
    Scan and aggregation iterate over runs, not individual values.
    """

    def __init__(self, values: List[Any]) -> None:
        self._runs: List[Tuple[Any, int]] = []  # (value, run_length)
        self._length = len(values)

        if not values:
            return

        current_val = values[0]
        current_run = 1
        for v in values[1:]:
            if v == current_val:
                current_run += 1
            else:
                self._runs.append((current_val, current_run))
                current_val = v
                current_run = 1
        self._runs.append((current_val, current_run))

    def to_list(self, mask: Optional[Mask] = None) -> List[Any]:
        expanded = []
        for val, count in self._runs:
            expanded.extend([val] * count)
        if mask is None:
            return expanded
        return [expanded[i] for i in mask.indices()]

    def sum(self) -> Any:
        return sum(val * count for val, count in self._runs)

    def __len__(self) -> int:
        return self._length


# ---------------------------------------------------------------------------
# Column Store
# ---------------------------------------------------------------------------

class ColumnStore:
    """
    Main columnar data structure.

    Data is stored per-column in plain Python lists. Each row is an index
    position across all columns — there is no row object.

    Analytical operations:
    - filter(): produce a Mask (bitmap), never materialize intermediate rows
    - aggregate(): apply an aggregation function over masked rows
    - groupby(): hash-based group aggregation
    - select(): materialize a subset of columns + masked rows into row dicts
    """

    def __init__(self) -> None:
        self._columns: Dict[str, List[Any]] = {}
        self._nrows: int = 0

    def add_column(self, name: str, values: List[Any]) -> None:
        """Add a column. All columns must have the same length."""
        if self._columns and len(values) != self._nrows:
            raise ValueError(
                f"Column '{name}' has {len(values)} rows, expected {self._nrows}"
            )
        self._columns[name] = list(values)
        self._nrows = len(values)

    def nrows(self) -> int:
        return self._nrows

    def ncols(self) -> int:
        return len(self._columns)

    def column_names(self) -> List[str]:
        return list(self._columns.keys())

    # ------------------------------------------------------------------
    # Filter (predicate pushdown)
    # ------------------------------------------------------------------

    _OPS: Dict[str, Callable[[Any, Any], bool]] = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }

    def filter(self, column: str, op: str, value: Any) -> Mask:
        """
        Vectorized column filter.

        column : column name
        op     : '==' | '!=' | '>' | '>=' | '<' | '<='
        value  : comparison value

        Returns a Mask (bitmap of matching rows).
        """
        if column not in self._columns:
            raise KeyError(f"Unknown column '{column}'")
        if op not in self._OPS:
            raise ValueError(f"Unknown operator '{op}'. Use: {list(self._OPS)}")

        fn = self._OPS[op]
        col = self._columns[column]
        return Mask([fn(v, value) for v in col])

    def filter_and(self, *conditions: Tuple[str, str, Any]) -> Mask:
        """
        Combine multiple filters with AND.
        conditions: sequence of (column, op, value) tuples.
        """
        mask = Mask.all_true(self._nrows)
        for col, op, val in conditions:
            mask = mask & self.filter(col, op, val)
        return mask

    def filter_or(self, *conditions: Tuple[str, str, Any]) -> Mask:
        """Combine multiple filters with OR."""
        mask = Mask.all_false(self._nrows)
        for col, op, val in conditions:
            mask = mask | self.filter(col, op, val)
        return mask

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    _AGG_FNS: Dict[str, Callable[[Iterable[Any]], Any]] = {
        "sum":   sum,
        "count": lambda xs: sum(1 for _ in xs),
        "min":   min,
        "max":   max,
        "avg":   lambda xs: (lambda lst: sum(lst) / len(lst) if lst else None)(list(xs)),
    }

    def aggregate(
        self,
        column: str,
        func: str,
        mask: Optional[Mask] = None,
    ) -> Any:
        """
        Aggregate a column.

        column : column name
        func   : 'sum' | 'count' | 'min' | 'max' | 'avg'
        mask   : row filter (None = all rows)
        """
        if column not in self._columns:
            raise KeyError(f"Unknown column '{column}'")
        if func not in self._AGG_FNS:
            raise ValueError(f"Unknown function '{func}'. Use: {list(self._AGG_FNS)}")

        col = self._columns[column]
        if mask is None:
            values = col
        else:
            values = [col[i] for i in mask.indices()]

        return self._AGG_FNS[func](values)

    # ------------------------------------------------------------------
    # Group By
    # ------------------------------------------------------------------

    def groupby(
        self,
        group_col: str,
        agg_spec: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Hash-based GROUP BY aggregation.

        group_col : column to group by
        agg_spec  : {column_name: agg_function} — what to compute per group

        Returns list of dicts, one per group.

        Example:
            store.groupby("city", {"sales": "sum", "qty": "avg"})
            → [{"city": "Paris", "sales": 250, "qty": 1.0},
               {"city": "London", "sales": 450, "qty": 2.0}, ...]
        """
        if group_col not in self._columns:
            raise KeyError(f"Unknown group column '{group_col}'")
        for col in agg_spec:
            if col not in self._columns:
                raise KeyError(f"Unknown agg column '{col}'")

        # Build groups: group_key → list of row indices
        groups: Dict[Any, List[int]] = {}
        for i, key in enumerate(self._columns[group_col]):
            groups.setdefault(key, []).append(i)

        results = []
        for group_key, row_indices in sorted(groups.items(), key=lambda x: str(x[0])):
            row: Dict[str, Any] = {group_col: group_key}
            for col_name, func in agg_spec.items():
                col = self._columns[col_name]
                values = [col[i] for i in row_indices]
                row[col_name] = self._AGG_FNS[func](values)
            results.append(row)

        return results

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def select(
        self,
        columns: Optional[List[str]] = None,
        mask: Optional[Mask] = None,
    ) -> List[Dict[str, Any]]:
        """
        Materialize rows as a list of dicts (for display / export).

        columns : which columns to include (None = all)
        mask    : row filter (None = all rows)

        Note: in a real system you avoid materializing until absolutely
        necessary (e.g. output to user). Keep everything as masks + column refs.
        """
        cols = columns if columns is not None else list(self._columns.keys())
        for c in cols:
            if c not in self._columns:
                raise KeyError(f"Unknown column '{c}'")

        indices = mask.indices() if mask is not None else range(self._nrows)
        return [{c: self._columns[c][i] for c in cols} for i in indices]

    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return first n rows as dicts."""
        return self.select(mask=Mask([i < n for i in range(self._nrows)]))

    def describe(self) -> Dict[str, Dict[str, Any]]:
        """Basic statistics for numeric columns."""
        result = {}
        for col_name, col in self._columns.items():
            try:
                result[col_name] = {
                    "count": len(col),
                    "sum": sum(col),
                    "min": min(col),
                    "max": max(col),
                    "avg": sum(col) / len(col),
                }
            except TypeError:
                result[col_name] = {"count": len(col), "type": "non-numeric"}
        return result
