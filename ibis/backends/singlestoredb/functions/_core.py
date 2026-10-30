"""Helpers for calling SingleStoreDB builtin functions."""

from __future__ import annotations

import functools
import inspect
from typing import Any

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir


@functools.cache
def _builtin(name: str, nargs: int, dtype: dt.DataType, aggregate: bool = False):
    def fn(*args): ...

    fn.__name__ = name.lower()
    fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter(f"arg{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for i in range(nargs)
        ]
    )
    decorator = ibis.udf.agg.builtin if aggregate else ibis.udf.scalar.builtin
    return decorator(fn, name=name, signature=((None,) * nargs, dtype))


def call(name: str, dtype: dt.DataType, *args: Any) -> ir.Value:
    """Call the scalar builtin `name`, which returns `dtype`."""
    return _builtin(name, len(args), dt.dtype(dtype))(*args)


def aggregate(name: str, dtype: dt.DataType, *args: Any, where=None) -> ir.Value:
    """Call the aggregate builtin `name`, which returns `dtype`."""
    return _builtin(name, len(args), dt.dtype(dtype), aggregate=True)(
        *args, where=where
    )
