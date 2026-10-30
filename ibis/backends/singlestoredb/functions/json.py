"""SingleStoreDB JSON functions.

Functions that take a key path accept any number of keys after the JSON
value: object keys as strings and zero-based array positions as integers,
as in `json_extract_string(t.doc, "address", "city")`.
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
    "json_agg",
    "json_array_contains_double",
    "json_array_contains_json",
    "json_array_contains_string",
    "json_array_push_double",
    "json_array_push_json",
    "json_array_push_string",
    "json_build_array",
    "json_build_object",
    "json_delete_key",
    "json_exclude_mask",
    "json_extract_bigint",
    "json_extract_double",
    "json_extract_json",
    "json_extract_string",
    "json_get_type",
    "json_include_mask",
    "json_keys",
    "json_length",
    "json_match_any",
    "json_match_any_exists",
    "json_merge_patch",
    "json_pretty",
    "json_set_double",
    "json_set_json",
    "json_set_string",
    "json_splice_double",
    "json_splice_json",
    "json_splice_string",
    "reduce",
    "to_json",
]


def _keypath(keys: tuple) -> tuple:
    for key in keys:
        if not isinstance(key, (str, int, ir.Value)):
            raise TypeError(f"keys must be strings or integers, got {key!r}")
    return keys


def json_agg(arg, /) -> ir.JSONValue:
    """Aggregate values into a JSON array, including NULLs.

    Filter the table first to aggregate a subset of rows; a `where`
    argument would add NULLs for the rows it excludes.
    """
    return aggregate("JSON_AGG", dt.json, arg)


def json_array_contains_string(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a string."""
    return call("JSON_ARRAY_CONTAINS_STRING", dt.boolean, array, value)


def json_array_contains_double(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a number."""
    return call("JSON_ARRAY_CONTAINS_DOUBLE", dt.boolean, array, value)


def json_array_contains_json(array, value, /) -> ir.BooleanValue:
    """Return whether a JSON array contains a JSON value."""
    return call("JSON_ARRAY_CONTAINS_JSON", dt.boolean, array, value)


def json_array_push_string(array, value, /) -> ir.JSONValue:
    """Append a string to a JSON array."""
    return call("JSON_ARRAY_PUSH_STRING", dt.json, array, value)


def json_array_push_double(array, value, /) -> ir.JSONValue:
    """Append a number to a JSON array."""
    return call("JSON_ARRAY_PUSH_DOUBLE", dt.json, array, value)


def json_array_push_json(array, value, /) -> ir.JSONValue:
    """Append a JSON value to a JSON array."""
    return call("JSON_ARRAY_PUSH_JSON", dt.json, array, value)


def json_build_array(*values) -> ir.JSONValue:
    """Return a JSON array of `values`."""
    return call("JSON_BUILD_ARRAY", dt.json, *values)


def json_build_object(*args) -> ir.JSONValue:
    """Return a JSON object from a mapping, or from alternating keys and values.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json_build_object({"name": t.name, "price": t.price})  # doctest: +SKIP
    >>> s2.json_build_object("name", t.name, "price", t.price)  # doctest: +SKIP
    """
    if len(args) == 1 and isinstance(args[0], Mapping):
        args = tuple(item for pair in args[0].items() for item in pair)
    elif len(args) % 2:
        raise TypeError("json_build_object takes alternating keys and values")
    return call("JSON_BUILD_OBJECT", dt.json, *args)


def json_delete_key(json_value, *keys) -> ir.JSONValue:
    """Return a JSON value with the value at the key path removed."""
    if not keys:
        raise TypeError("json_delete_key takes at least one key")
    return call("JSON_DELETE_KEY", dt.json, json_value, *_keypath(keys))


def json_extract_json(json_value, *keys) -> ir.JSONValue:
    """Return the JSON value at the key path."""
    return call("JSON_EXTRACT_JSON", dt.json, json_value, *_keypath(keys))


def json_extract_string(json_value, *keys) -> ir.StringValue:
    """Return the value at the key path as a string.

    Strings are unquoted, and other values give their JSON text.
    """
    return call("JSON_EXTRACT_STRING", dt.string, json_value, *_keypath(keys))


def json_extract_double(json_value, *keys) -> ir.FloatingValue:
    """Return the value at the key path as a float."""
    return call("JSON_EXTRACT_DOUBLE", dt.float64, json_value, *_keypath(keys))


def json_extract_bigint(json_value, *keys) -> ir.IntegerValue:
    """Return the value at the key path as an integer, rounding floats."""
    return call("JSON_EXTRACT_BIGINT", dt.int64, json_value, *_keypath(keys))


def json_get_type(json_value, /) -> ir.StringValue:
    """Return the JSON type of a value, such as `"object"` or `"array"`."""
    return call("JSON_GET_TYPE", dt.string, json_value)


def json_include_mask(json_value, mask, /) -> ir.JSONValue:
    """Return the parts of a JSON object selected by a mask such as `{"a": 1}`."""
    return call("JSON_INCLUDE_MASK", dt.json, json_value, mask)


def json_exclude_mask(json_value, mask, /) -> ir.JSONValue:
    """Return a JSON object without the parts selected by a mask."""
    return call("JSON_EXCLUDE_MASK", dt.json, json_value, mask)


def json_keys(json_value, *keys) -> ir.JSONValue:
    """Return the keys of the JSON object at the key path as a JSON array."""
    return call("JSON_KEYS", dt.json, json_value, *_keypath(keys))


def json_length(json_value, /) -> ir.IntegerValue:
    """Return the number of elements in a JSON array or keys in an object."""
    return call("JSON_LENGTH", dt.int64, json_value)


_MATCH_PARAMS = {
    "bigint": ("MATCH_PARAM_BIGINT_STRICT", dt.int64),
    "double": ("MATCH_PARAM_DOUBLE_STRICT", dt.float64),
    "string": ("MATCH_PARAM_STRING_STRICT", dt.string),
    "bool": ("MATCH_PARAM_BOOL_STRICT", dt.boolean),
    "json": ("MATCH_PARAM_JSON", dt.json),
}


def json_match_any(
    predicate: Callable[[ir.Value], ir.BooleanValue],
    json_value,
    *keys,
    value_type: Literal["bigint", "double", "string", "bool", "json"],
) -> ir.BooleanValue:
    """Return whether any value at the key path satisfies `predicate`.

    The key path matches through arrays, so `"a"` matches each element of an
    array at `a`. `predicate` is called with an expression for each value
    of type `value_type`; values of other types don't match.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json_match_any(
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
    return call("JSON_MATCH_ANY", dt.boolean, condition, json_value, *_keypath(keys))


def json_match_any_exists(json_value, *keys) -> ir.BooleanValue:
    """Return whether any value, possibly null, exists at the key path.

    The key path matches through arrays, like `json_match_any`.
    """
    return call("JSON_MATCH_ANY_EXISTS", dt.boolean, json_value, *_keypath(keys))


def json_merge_patch(json_value, patch, /) -> ir.JSONValue:
    """Merge a JSON patch into a JSON value, removing keys whose value is null."""
    return call("JSON_MERGE_PATCH", dt.json, json_value, patch)


def json_pretty(json_value, /) -> ir.StringValue:
    """Return indented JSON text."""
    return call("JSON_PRETTY", dt.string, json_value)


def _json_set(name: str, json_value, args: tuple) -> ir.JSONValue:
    if len(args) < 2:
        raise TypeError(f"{name.lower()} takes at least one key and a value")
    *keys, value = args
    return call(name, dt.json, json_value, *_keypath(tuple(keys)), value)


def json_set_string(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a string set at the key path.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.json_set_string(t.doc, "address", "city", "Boston")  # doctest: +SKIP
    """
    return _json_set("JSON_SET_STRING", json_value, keys_and_value)


def json_set_double(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a number set at the key path."""
    return _json_set("JSON_SET_DOUBLE", json_value, keys_and_value)


def json_set_json(json_value, *keys_and_value) -> ir.JSONValue:
    """Return a JSON value with a JSON value set at the key path."""
    return _json_set("JSON_SET_JSON", json_value, keys_and_value)


def json_splice_string(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with strings."""
    return call("JSON_SPLICE_STRING", dt.json, array, start, length, *values)


def json_splice_double(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with numbers."""
    return call("JSON_SPLICE_DOUBLE", dt.json, array, start, length, *values)


def json_splice_json(array, start, length, /, *values) -> ir.JSONValue:
    """Replace `length` elements of a JSON array from `start` with JSON values."""
    return call("JSON_SPLICE_JSON", dt.json, array, start, length, *values)


def to_json(value, /) -> ir.JSONValue:
    """Convert a value to JSON."""
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

    `fn` is called with expressions for the accumulator, which starts as
    `initial`, and the current element, converted to `value_type`. If `fn`
    takes a third argument, it gets the element's zero-based position. Its
    result is the next accumulator.

    Examples
    --------
    >>> from ibis.backends.singlestoredb import functions as s2
    >>> s2.reduce(
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
