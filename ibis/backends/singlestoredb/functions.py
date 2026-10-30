"""SingleStoreDB functions for `con.function`, and builtin function signatures."""

from __future__ import annotations

import functools
import inspect
import json
from typing import TYPE_CHECKING, Any, NamedTuple

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

if TYPE_CHECKING:
    from collections.abc import Callable

    from ibis.backends.singlestoredb import Backend

# Element types of packed vectors, selected by a function name suffix such as
# DOT_PRODUCT_I8. Without a suffix, packed vectors hold 32-bit floats.
ELEMENT_TYPES = {
    "i8": dt.int8,
    "i16": dt.int16,
    "i32": dt.int32,
    "i64": dt.int64,
    "f32": dt.float32,
    "f64": dt.float64,
}

# Return types that depend on the vector arguments
VECTOR = "vector"  # the type of the vector arguments
ELEMENT = "element"  # the type of one element
ACCUMULATOR = "accumulator"  # int64 for integer elements, float64 otherwise


class Builtin(NamedTuple):
    returns: dt.DataType | str
    min_args: int
    max_args: int | None  # None for variadic functions
    vectors: tuple[int, ...] = ()  # positions of the vector arguments
    aggregate: bool = False


# Functions that accept an element type suffix
VECTOR_FUNCTIONS = {
    "json_array_pack": Builtin(dt.binary, 1, 1),
    "json_array_unpack": Builtin(dt.json, 1, 1),
    "dot_product": Builtin(ACCUMULATOR, 2, 2, (0, 1)),
    "euclidean_distance": Builtin(dt.float64, 2, 2, (0, 1)),
    "vector_add": Builtin(VECTOR, 2, 2, (0, 1)),
    "vector_sub": Builtin(VECTOR, 2, 2, (0, 1)),
    "vector_mul": Builtin(VECTOR, 2, 2, (0, 1)),
    "scalar_vector_mul": Builtin(VECTOR, 2, 2, (1,)),
    "vector_num_elements": Builtin(dt.int64, 1, 1, (0,)),
    "vector_elements_sum": Builtin(ACCUMULATOR, 1, 1, (0,)),
    "vector_kth_element": Builtin(ELEMENT, 2, 2, (0,)),
    "vector_sort": Builtin(VECTOR, 1, 2, (0,)),
    "vector_subvector": Builtin(VECTOR, 3, 3, (0,)),
    "vector_sum": Builtin(VECTOR, 1, 1, (0,), aggregate=True),
}

OTHER_FUNCTIONS = {
    "json_get_type": Builtin(dt.string, 1, 1),
    "json_extract_string": Builtin(dt.string, 1, None),
    "json_extract_double": Builtin(dt.float64, 1, None),
    "json_extract_bigint": Builtin(dt.int64, 1, None),
    "json_extract_json": Builtin(dt.json, 1, None),
    "json_keys": Builtin(dt.json, 1, None),
    "json_length": Builtin(dt.int64, 1, 1),
    "json_delete_key": Builtin(dt.json, 2, None),
    "json_set_string": Builtin(dt.json, 3, None),
    "json_set_double": Builtin(dt.json, 3, None),
    "json_set_json": Builtin(dt.json, 3, None),
    "json_array_contains_string": Builtin(dt.boolean, 2, 2),
    "json_array_contains_double": Builtin(dt.boolean, 2, 2),
    "json_array_contains_json": Builtin(dt.boolean, 2, 2),
    "json_array_push_string": Builtin(dt.json, 2, 2),
    "json_array_push_double": Builtin(dt.json, 2, 2),
    "json_array_push_json": Builtin(dt.json, 2, 2),
    "json_splice_string": Builtin(dt.json, 3, None),
    "json_splice_double": Builtin(dt.json, 3, None),
    "json_splice_json": Builtin(dt.json, 3, None),
    "json_pretty": Builtin(dt.string, 1, 1),
    "json_include_mask": Builtin(dt.json, 2, 2),
    "json_exclude_mask": Builtin(dt.json, 2, 2),
}


class Functions:
    """Callables for SingleStoreDB functions, by attribute or by name."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def __call__(self, name: str, *, database: str | None = None) -> Callable:
        """Return a callable for a SingleStoreDB function.

        `name` can be a builtin vector or JSON function that has no Ibis
        method, such as `dot_product` or `json_get_type`, or a user-defined
        function or aggregate in `database`.

        Vector arguments of builtins can be `VECTOR` columns, `BLOB` columns
        of packed vectors, or Python sequences. Use a suffix such as
        `dot_product_i8` for `BLOB` vectors of other element types. Functions
        that return a vector return a `VECTOR` if any vector argument is
        one, and a packed `BLOB` otherwise.

        Parameters
        ----------
        name
            The function name.
        database
            The database of a user-defined function. Defaults to the current
            database. Builtins are only looked up when `database` is None.

        Returns
        -------
        Callable
            A function that returns an Ibis expression.
        """
        if database is None and (builtin := lookup(name)) is not None:
            return builtin
        return self._backend._udf(name, database=database)

    def __getattr__(self, name: str) -> Callable:
        # Fail fast for dunder and private probes, e.g. from IPython or copy
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self(name)
        except com.MissingUDFError as e:
            raise AttributeError(f"no SingleStoreDB function {name!r}") from e

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *names()})


def names() -> list[str]:
    """Return the names of the builtin functions, without element type suffixes."""
    return [*VECTOR_FUNCTIONS, *OTHER_FUNCTIONS]


def lookup(name: str) -> Callable | None:
    """Return a callable for the builtin function `name`, or None."""
    return _lookup(name.lower())


@functools.cache
def _lookup(name: str) -> Callable | None:
    base, _, suffix = name.rpartition("_")
    if suffix in ELEMENT_TYPES and base in VECTOR_FUNCTIONS:
        return _function(base, VECTOR_FUNCTIONS[base], suffix)
    if (spec := VECTOR_FUNCTIONS.get(name, OTHER_FUNCTIONS.get(name))) is not None:
        return _function(name, spec, None)
    return None


@functools.cache
def _builtin_function(name: str, nargs: int, dtype: dt.DataType, aggregate: bool):
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


def _function(name: str, spec: Builtin, suffix: str | None) -> Callable:
    sql_name = name.upper() if suffix is None else f"{name}_{suffix}".upper()

    def fn(*args: Any, **kwargs: Any) -> ir.Value:
        nargs = len(args)
        if nargs < spec.min_args or (
            spec.max_args is not None and nargs > spec.max_args
        ):
            if spec.max_args is None:
                expected = f"at least {spec.min_args}"
            elif spec.min_args == spec.max_args:
                expected = str(spec.min_args)
            else:
                expected = f"{spec.min_args} to {spec.max_args}"
            raise TypeError(
                f"{sql_name} takes {expected} arguments, but {nargs} were given"
            )
        args, native = _vector_args(args, spec.vectors, suffix)
        dtype = _return_type(spec.returns, native, suffix)
        return _builtin_function(sql_name, nargs, dtype, spec.aggregate)(
            *args, **kwargs
        )

    fn.__name__ = sql_name.lower()
    fn.__doc__ = f"Call the SingleStoreDB builtin function `{sql_name}`."
    return fn


def _vector_args(args, positions, suffix):
    """Convert Python sequences to vectors and find any `VECTOR` argument type.

    `VECTOR` values are fixed-length arrays; packed vectors are `BLOB`s.
    """
    args = list(args)
    vectors = [args[i] for i in positions if i < len(args)]
    native = next(
        (
            arg.type()
            for arg in vectors
            if isinstance(arg, ir.Value) and arg.type().is_array()
        ),
        None,
    )
    for i in positions:
        if i >= len(args) or isinstance(args[i], ir.Value):
            continue
        value = args[i]
        if not isinstance(value, str):
            # `.item()` converts numpy scalars to Python numbers
            value = json.dumps(list(value), default=lambda x: x.item())
        if native is not None:
            # SingleStoreDB converts JSON array strings to the other argument's
            # VECTOR type, including its element type
            args[i] = ibis.literal(value)
        else:
            pack = "JSON_ARRAY_PACK" if suffix is None else f"JSON_ARRAY_PACK_{suffix}"
            args[i] = _builtin_function(pack.upper(), 1, dt.binary, False)(value)
    return args, native


def _return_type(returns, native, suffix):
    if isinstance(returns, dt.DataType):
        return returns
    if returns == VECTOR:
        # Mixing a VECTOR with a packed vector gives a VECTOR
        return dt.binary if native is None else native
    if native is not None and suffix is None:
        element = native.value_type.copy(nullable=True)
    else:
        element = ELEMENT_TYPES[suffix or "f32"]
    if returns == ELEMENT:
        return element
    return dt.int64 if element.is_integer() else dt.float64
