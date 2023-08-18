import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value
from ibis.common.typing import VarTuple


class JSONSetDouble(Value):
    """JSON set double value."""

    arg = Value[dt.JSON | dt.String]
    # TODO: min_length=1
    key_path = VarTuple[Value[dt.String | dt.Integer]]
    value = dt.Float64

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_set_double(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a double value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_value : strings or ints and double
        Keys or zero-indexed array positions followed by the double value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetDouble(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_double = json_set_double
ir.StringValue.json_set_double = json_set_double


class JSONSetString(Value):
    """JSON set string value."""

    arg = Value[dt.JSON | dt.String]
    # TODO: min_length=1
    key_path = VarTuple[Value[dt.String | dt.Integer]]
    value = dt.String

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_set_string(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a double value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_value : strings or ints
        Keys or zero-indexed array positions followed by the string value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetString(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_string = json_set_string
ir.StringValue.json_set_string = json_set_string


class JSONSetJSON(Value):
    """JSON set JSON value."""

    arg = Value[dt.JSON | dt.String]
    # TODO: min_length=1
    key_path = VarTuple[Value[dt.String | dt.Integer]]
    value = dt.String

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_set_json(
    arg: ir.JSONValue | ir.StringValue,
    *key_value: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Set a JSON value at the specified key path.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_value : strings or ints
        Keys or zero-indexed array positions followed by the JSON value

    Returns
    -------
    String or JSON Column

    """
    return JSONSetJSON(arg, key_value[:-1], key_value[-1]).to_expr()


ir.JSONValue.json_set_json = json_set_json
ir.StringValue.json_set_json = json_set_json
