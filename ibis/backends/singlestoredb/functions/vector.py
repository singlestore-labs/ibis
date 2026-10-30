"""SingleStoreDB vector functions.

Vector arguments can be `VECTOR` columns, JSON arrays (array expressions
or JSON array strings), Python sequences, or `BLOB` columns of packed
vectors.

The element type is taken from a `VECTOR` argument when there is one; the
other arguments are converted to its type. Otherwise arguments are packed
with the element type of an array argument, such as `array<int8>`, or as
32-bit floats. `BLOB` vectors don't record their element type, so pass
`element_type` for packed vectors whose elements aren't 32-bit floats;
without it, their bytes are read as 32-bit floats.

Functions that return a vector return a `VECTOR` if any argument is one,
and a packed `BLOB` otherwise.
"""

from __future__ import annotations

import json
from typing import Literal

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.backends.singlestoredb.functions._core import aggregate, call

__all__ = [
    "add",
    "dot_product",
    "elements_sum",
    "euclidean_distance",
    "from_json",
    "kth_element",
    "mul",
    "num_elements",
    "scalar_mul",
    "sort",
    "sub",
    "subvector",
    "sum",
    "to_json",
]

ElementType = Literal["i8", "i16", "i32", "i64", "f32", "f64"]

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


def _is_vector(arg) -> bool:
    # VECTOR columns are read as fixed-length arrays
    return isinstance(arg, ir.Value) and arg.type().is_fixed_length_array()


def _as_json_array(value) -> str:
    if isinstance(value, str):
        return value
    # `.item()` converts numpy scalars to Python numbers
    return json.dumps(list(value), default=lambda x: x.item())


def _array_element_type(arg) -> str | None:
    if not (isinstance(arg, ir.Value) and arg.type().is_array()):
        return None
    value_type = arg.type().value_type.copy(nullable=True)
    return next(
        (name for name, typ in ELEMENT_TYPES.items() if typ == value_type), None
    )


class _Vectors:
    """Vector arguments converted to one representation."""

    def __init__(self, args, element_type: ElementType | None):
        if element_type is not None and element_type not in ELEMENT_TYPES:
            raise ValueError(
                f"element_type must be one of {sorted(ELEMENT_TYPES)}, "
                f"got {element_type!r}"
            )
        self.native = next((arg.type() for arg in args if _is_vector(arg)), None)
        if self.native is not None:
            native_element = self.native.value_type.copy(nullable=True)
            if (
                element_type is not None
                and ELEMENT_TYPES[element_type] != native_element
            ):
                raise ValueError(
                    f"element_type={element_type!r} doesn't match the "
                    f"{self.native} argument"
                )
            # SingleStoreDB converts JSON arrays and packed vectors to the
            # VECTOR argument's type, and rejects suffixed function names
            self.suffix = None
            self.element = native_element
            self.args = [
                arg if isinstance(arg, ir.Value) else ibis.literal(_as_json_array(arg))
                for arg in args
            ]
        else:
            if element_type is None:
                element_type = next(filter(None, map(_array_element_type, args)), "f32")
            self.suffix = None if element_type == "f32" else element_type
            self.element = ELEMENT_TYPES[element_type]
            self.args = [self._pack(arg) for arg in args]

    def _pack(self, arg):
        if isinstance(arg, ir.Value) and arg.type().is_binary():
            return arg
        if not isinstance(arg, ir.Value):
            arg = _as_json_array(arg)
        return call(self.name("JSON_ARRAY_PACK"), dt.binary, arg)

    def name(self, name: str) -> str:
        return name if self.suffix is None else f"{name}_{self.suffix.upper()}"

    @property
    def vector(self) -> dt.DataType:
        return dt.binary if self.native is None else self.native

    @property
    def accumulator(self) -> dt.DataType:
        return dt.int64 if self.element.is_integer() else dt.float64


def dot_product(
    left, right, /, *, element_type: ElementType | None = None
) -> ir.NumericValue:
    """Return the dot product of two vectors.

    Calls `DOT_PRODUCT`.

    The result is an integer for integer vectors and a float otherwise.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> t.select(score=s2.vector.dot_product(t.embedding, [0.1, 0.2, 0.3]))  # doctest: +SKIP
    """
    v = _Vectors([left, right], element_type)
    return call(v.name("DOT_PRODUCT"), v.accumulator, *v.args)


def euclidean_distance(
    left, right, /, *, element_type: ElementType | None = None
) -> ir.FloatingValue:
    """Return the Euclidean distance between two vectors.

    Calls `EUCLIDEAN_DISTANCE`.
    """
    v = _Vectors([left, right], element_type)
    return call(v.name("EUCLIDEAN_DISTANCE"), dt.float64, *v.args)


def add(left, right, /, *, element_type: ElementType | None = None) -> ir.Value:
    """Return the element-wise sum of two vectors.

    Calls `VECTOR_ADD`.
    """
    v = _Vectors([left, right], element_type)
    return call(v.name("VECTOR_ADD"), v.vector, *v.args)


def sub(left, right, /, *, element_type: ElementType | None = None) -> ir.Value:
    """Return the element-wise difference of two vectors.

    Calls `VECTOR_SUB`.
    """
    v = _Vectors([left, right], element_type)
    return call(v.name("VECTOR_SUB"), v.vector, *v.args)


def mul(left, right, /, *, element_type: ElementType | None = None) -> ir.Value:
    """Return the element-wise product of two vectors.

    Calls `VECTOR_MUL`.
    """
    v = _Vectors([left, right], element_type)
    return call(v.name("VECTOR_MUL"), v.vector, *v.args)


def scalar_mul(
    scalar, vector, /, *, element_type: ElementType | None = None
) -> ir.Value:
    """Return a vector with each element multiplied by `scalar`.

    Calls `SCALAR_VECTOR_MUL`.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("SCALAR_VECTOR_MUL"), v.vector, scalar, *v.args)


def num_elements(
    vector, /, *, element_type: ElementType | None = None
) -> ir.IntegerValue:
    """Return the number of elements in a vector.

    Calls `VECTOR_NUM_ELEMENTS`.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("VECTOR_NUM_ELEMENTS"), dt.int64, *v.args)


def elements_sum(
    vector, /, *, element_type: ElementType | None = None
) -> ir.NumericValue:
    """Return the sum of a vector's elements.

    Calls `VECTOR_ELEMENTS_SUM`.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("VECTOR_ELEMENTS_SUM"), v.accumulator, *v.args)


def kth_element(
    vector, k, /, *, element_type: ElementType | None = None
) -> ir.NumericValue:
    """Return the element at zero-based position `k` of a vector.

    Calls `VECTOR_KTH_ELEMENT`.

    Negative positions count from the end. SingleStoreDB raises an error for
    positions out of range.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("VECTOR_KTH_ELEMENT"), v.element, *v.args, k)


def sort(
    vector,
    direction: Literal["asc", "desc"] | None = None,
    /,
    *,
    element_type: ElementType | None = None,
) -> ir.Value:
    """Return a vector with its elements sorted, ascending by default.

    Calls `VECTOR_SORT`.
    """
    v = _Vectors([vector], element_type)
    args = v.args if direction is None else [*v.args, direction]
    return call(v.name("VECTOR_SORT"), v.vector, *args)


def subvector(
    vector, start, length, /, *, element_type: ElementType | None = None
) -> ir.Value:
    """Return `length` elements of a vector from zero-based position `start`.

    Calls `VECTOR_SUBVECTOR`.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("VECTOR_SUBVECTOR"), v.vector, *v.args, start, length)


def sum(vector, /, *, element_type: ElementType | None = None, where=None) -> ir.Value:
    """Aggregate vectors into their element-wise sum.

    Calls `VECTOR_SUM`.
    """
    v = _Vectors([vector], element_type)
    return aggregate(v.name("VECTOR_SUM"), v.vector, *v.args, where=where)


def from_json(array, /, *, element_type: ElementType | None = None) -> ir.Value:
    """Pack a JSON array of numbers into a `BLOB` vector.

    Calls `JSON_ARRAY_PACK`.

    `array` can be a JSON array expression, a JSON array string, or a Python
    sequence. The element type defaults to that of an array expression, or
    to 32-bit floats.
    """
    if _is_vector(array):
        raise TypeError("from_json takes a JSON array, not a VECTOR")
    return _Vectors([array], element_type).args[0]


def to_json(vector, /, *, element_type: ElementType | None = None) -> ir.JSONValue:
    """Unpack a `BLOB` vector into a JSON array of numbers.

    Calls `JSON_ARRAY_UNPACK`.
    """
    v = _Vectors([vector], element_type)
    return call(v.name("JSON_ARRAY_UNPACK"), dt.json, *v.args)
