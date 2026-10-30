"""SingleStoreDB JSON functions.

Functions that take a key path accept any number of keys after the JSON
value: object keys as strings and zero-based array positions as integers,
as in `extract_string(t.doc, "address", "city")`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Literal

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.backends.singlestoredb.functions._core import aggregate, call

__all__ = [
    "agg",
    "array_contains_double",
    "array_contains_json",
    "array_contains_string",
    "array_push_double",
    "array_push_json",
    "array_push_string",
    "build_array",
    "build_object",
    "delete_key",
    "exclude_mask",
    "extract_bigint",
    "extract_double",
    "extract_json",
    "extract_string",
    "from_value",
    "get_type",
    "include_mask",
    "keys",
    "length",
    "match_any",
    "match_any_exists",
    "merge_patch",
    "pretty",
    "reduce",
    "set_double",
    "set_json",
    "set_string",
    "splice_double",
    "splice_json",
    "splice_string",
]


def _keypath(keys: tuple) -> tuple:
    for key in keys:
        if not isinstance(key, (str, int, ir.Value)):
            raise TypeError(f"keys must be strings or integers, got {key!r}")
    return keys


def agg(arg, /) -> ir.JSONValue:
    """Aggregate values into a JSON array, including NULLs.

    Calls `JSON_AGG`.

    Filter the table first to aggregate a subset of rows; a `where`
    argument would add NULLs for the rows it excludes.
    """
    return aggregate("JSON_AGG", dt.json, arg)


def array_contains_string(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a string.

    Calls `JSON_ARRAY_CONTAINS_STRING`.
    """
    return call("JSON_ARRAY_CONTAINS_STRING", dt.boolean, array, value)


def array_contains_double(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a number.

    Calls `JSON_ARRAY_CONTAINS_DOUBLE`.
    """
    return call("JSON_ARRAY_CONTAINS_DOUBLE", dt.boolean, array, value)


def array_contains_json(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a JSON value.

    Calls `JSON_ARRAY_CONTAINS_JSON`.
    """
    return call("JSON_ARRAY_CONTAINS_JSON", dt.boolean, array, value)


def array_push_string(array, value, /) -> ir.JSONValue:
    """Append a string to a JSON array.

    Calls `JSON_ARRAY_PUSH_STRING`.
    """
    return call("JSON_ARRAY_PUSH_STRING", dt.json, array, value)


def array_push_double(array, value, /) -> ir.JSONValue:
    """Append a number to a JSON array.

    Calls `JSON_ARRAY_PUSH_DOUBLE`.
    """
    return call("JSON_ARRAY_PUSH_DOUBLE", dt.json, array, value)


def array_push_json(array, value, /) -> ir.JSONValue:
    """Append a JSON value to a JSON array.

    Calls `JSON_ARRAY_PUSH_JSON`.
    """
    return call("JSON_ARRAY_PUSH_JSON", dt.json, array, value)


def build_array(*values) -> ir.JSONValue:
    """Return a JSON array of `values`.

    Calls `JSON_BUILD_ARRAY`.
    """
    return call("JSON_BUILD_ARRAY", dt.json, *values)


def build_object(*args) -> ir.JSONValue:
    """Return a JSON object from a mapping, or from alternating keys and values.

    Calls `JSON_BUILD_OBJECT`.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json.build_object({"name": t.name, "price": t.price})  # doctest: +SKIP
    >>> s2.json.build_object("name", t.name, "price", t.price)  # doctest: +SKIP
    """
    if len(args) == 1 and isinstance(args[0], Mapping):
        args = tuple(item for pair in args[0].items() for item in pair)
    elif len(args) % 2:
        raise TypeError("build_object takes alternating keys and values")
    return call("JSON_BUILD_OBJECT", dt.json, *args)


def delete_key(json_value, *path) -> ir.JSONValue:
    """Return a JSON value with the value at the key path removed.

    Calls `JSON_DELETE_KEY`.
    """
    if not path:
        raise TypeError("delete_key takes at least one key")
    return call("JSON_DELETE_KEY", dt.json, json_value, *_keypath(path))


def extract_json(json_value, *path) -> ir.JSONValue:
    """Return the JSON value at the key path.

    Calls `JSON_EXTRACT_JSON`.
    """
    return call("JSON_EXTRACT_JSON", dt.json, json_value, *_keypath(path))


def extract_string(json_value, *path) -> ir.StringValue:
    """Return the value at the key path as a string.

    Calls `JSON_EXTRACT_STRING`.

    Strings are unquoted, and other values give their JSON text.
    """
    return call("JSON_EXTRACT_STRING", dt.string, json_value, *_keypath(path))


def extract_double(json_value, *path) -> ir.FloatingValue:
    """Return the value at the key path as a float.

    Calls `JSON_EXTRACT_DOUBLE`.
    """
    return call("JSON_EXTRACT_DOUBLE", dt.float64, json_value, *_keypath(path))


def extract_bigint(json_value, *path) -> ir.IntegerValue:
    """Return the value at the key path as an integer, rounding floats.

    Calls `JSON_EXTRACT_BIGINT`.
    """
    return call("JSON_EXTRACT_BIGINT", dt.int64, json_value, *_keypath(path))


def get_type(json_value, /) -> ir.StringValue:
    """Return the JSON type of a value, such as `"object"` or `"array"`.

    Calls `JSON_GET_TYPE`.
    """
    return call("JSON_GET_TYPE", dt.string, json_value)


def include_mask(json_value, mask, /) -> ir.JSONValue:
    """Return the parts of a JSON object selected by a mask such as `{"a": 1}`.

    Calls `JSON_INCLUDE_MASK`.
    """
    return call("JSON_INCLUDE_MASK", dt.json, json_value, mask)


def exclude_mask(json_value, mask, /) -> ir.JSONValue:
    """Return a JSON object without the parts selected by a mask.

    Calls `JSON_EXCLUDE_MASK`.
    """
    return call("JSON_EXCLUDE_MASK", dt.json, json_value, mask)


def keys(json_value, *path) -> ir.JSONValue:
    """Return the keys of the JSON object at the key path as a JSON array.

    Calls `JSON_KEYS`.
    """
    return call("JSON_KEYS", dt.json, json_value, *_keypath(path))


def length(json_value, /) -> ir.IntegerValue:
    """Return the number of elements in a JSON array or keys in an object.

    Calls `JSON_LENGTH`.
    """
    return call("JSON_LENGTH", dt.int64, json_value)


_MATCH_PARAMS = {
    "bigint": ("MATCH_PARAM_BIGINT_STRICT", dt.int64),
    "double": ("MATCH_PARAM_DOUBLE_STRICT", dt.float64),
    "string": ("MATCH_PARAM_STRING_STRICT", dt.string),
    "bool": ("MATCH_PARAM_BOOL_STRICT", dt.boolean),
    "json": ("MATCH_PARAM_JSON", dt.json),
}


def match_any(
    predicate: Callable[[ir.Value], ir.BooleanValue],
    json_value,
    *path,
    value_type: Literal["bigint", "double", "string", "bool", "json"],
) -> ir.BooleanValue:
    """Return whether any value at the key path satisfies `predicate`.

    Calls `JSON_MATCH_ANY`.

    The key path matches through arrays, so `"a"` matches each element of an
    array at `a`. `predicate` is called with an expression for each value
    of type `value_type`; values of other types don't match.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json.match_any(
    ...     lambda v: v > 4, t.doc, "scores", value_type="bigint"
    ... )  # doctest: +SKIP
    """
    try:
        name, dtype = _MATCH_PARAMS[value_type]
    except KeyError:
        raise ValueError(
            f"value_type must be one of {sorted(_MATCH_PARAMS)}, got {value_type!r}"
        ) from None
    condition = predicate(call(name, dtype))
    return call("JSON_MATCH_ANY", dt.boolean, condition, json_value, *_keypath(path))


def match_any_exists(json_value, *path) -> ir.BooleanValue:
    """Return whether any value, possibly null, exists at the key path.

    Calls `JSON_MATCH_ANY_EXISTS`.

    The key path matches through arrays, like `match_any`.
    """
    return call("JSON_MATCH_ANY_EXISTS", dt.boolean, json_value, *_keypath(path))


def merge_patch(json_value, patch, /) -> ir.JSONValue:
    """Merge a JSON patch into a JSON value, removing keys whose value is null.

    Calls `JSON_MERGE_PATCH`.
    """
    return call("JSON_MERGE_PATCH", dt.json, json_value, patch)


def pretty(json_value, /) -> ir.StringValue:
    """Return indented JSON text.

    Calls `JSON_PRETTY`.
    """
    return call("JSON_PRETTY", dt.string, json_value)


def _json_set(name: str, json_value, args: tuple) -> ir.JSONValue:
    if len(args) < 2:
        raise TypeError(
            f"{name.lower().removeprefix('json_')} takes at least one key and a value"
        )
    *keys, value = args
    return call(name, dt.json, json_value, *_keypath(tuple(keys)), value)


def set_string(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a string set at the key path.

    Calls `JSON_SET_STRING`.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json.set_string(t.doc, "address", "city", "Boston")  # doctest: +SKIP
    """
    return _json_set("JSON_SET_STRING", json_value, keys_and_value)


def set_double(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a number set at the key path.

    Calls `JSON_SET_DOUBLE`.
    """
    return _json_set("JSON_SET_DOUBLE", json_value, keys_and_value)


def set_json(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a JSON value set at the key path.

    Calls `JSON_SET_JSON`.
    """
    return _json_set("JSON_SET_JSON", json_value, keys_and_value)


def splice_string(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with strings.

    Calls `JSON_SPLICE_STRING`.
    """
    return call("JSON_SPLICE_STRING", dt.json, array, start, length, *values)


def splice_double(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with numbers.

    Calls `JSON_SPLICE_DOUBLE`.
    """
    return call("JSON_SPLICE_DOUBLE", dt.json, array, start, length, *values)


def splice_json(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with JSON values.

    Calls `JSON_SPLICE_JSON`.
    """
    return call("JSON_SPLICE_JSON", dt.json, array, start, length, *values)


def from_value(value, /) -> ir.JSONValue:
    """Convert a value to JSON.

    Calls `TO_JSON`.

    Text becomes a JSON string; use `.cast("json")` to parse JSON text.
    """
    return call("TO_JSON", dt.json, value)


_REDUCE_VALUES = {
    "json": None,
    "bigint": ("JSON_EXTRACT_BIGINT", dt.int64),
    "double": ("JSON_EXTRACT_DOUBLE", dt.float64),
    "string": ("JSON_EXTRACT_STRING", dt.string),
}


def reduce(
    initial,
    array,
    fn: Callable[..., ir.Value],
    /,
    *,
    value_type: Literal["json", "bigint", "double", "string"] = "json",
) -> ir.Value:
    """Fold the elements of a JSON array into a value.

    Calls `REDUCE`.

    `fn` is called with expressions for the accumulator, which starts as
    `initial`, and the current element, converted to `value_type`. If `fn`
    takes a third argument, it gets the element's zero-based position. Its
    result is the next accumulator.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json.reduce(
    ...     0, t.doc["scores"], lambda acc, v: acc + v, value_type="bigint"
    ... )  # doctest: +SKIP
    """
    if value_type not in _REDUCE_VALUES:
        raise ValueError(
            f"value_type must be one of {sorted(_REDUCE_VALUES)}, got {value_type!r}"
        )
    if not isinstance(initial, ir.Value):
        initial = ibis.literal(initial)
    acc = call("REDUCE_ACC", initial.type())
    value = call("REDUCE_VALUE", dt.json)
    if (extract := _REDUCE_VALUES[value_type]) is not None:
        value = call(*extract, value)
    args = [acc, value]
    if len(inspect.signature(fn).parameters) > 2:
        args.append(call("REDUCE_INDEX", dt.int64))
    body = fn(*args)
    elements = call("JSON_TO_ARRAY", dt.json, array)
    return call("REDUCE", body.type(), initial, elements, body)
