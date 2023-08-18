import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value

StrictNumeric = Value[dt.Numeric]


class JSONArrayPushDouble(Value):
    """JSON array append."""

    arg = Value[dt.JSON | dt.String]
    value = StrictNumeric

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_array_push_double(
    arg: ir.JSONValue | ir.StringValue, value: ir.NumericValue
) -> ir.JSONValue | ir.StringValue:
    """
    Append a double to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : float
        Float value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushDouble(arg, value).to_expr()


ir.JSONValue.json_array_push_double = json_array_push_double
ir.StringValue.json_array_push_double = json_array_push_double


class JSONArrayPushString(Value):
    """JSON array append."""

    arg = Value[dt.JSON | dt.String]
    value = dt.String

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_array_push_string(
    arg: ir.JSONValue | ir.StringValue, value: ir.StringValue
) -> ir.JSONValue | ir.StringValue:
    """
    Append a string to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        String value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushString(arg, value).to_expr()


ir.JSONValue.json_array_push_string = json_array_push_string
ir.StringValue.json_array_push_string = json_array_push_string


class JSONArrayPushJSON(Value):
    """JSON array append."""

    arg = Value[dt.JSON | dt.String]
    value = Value[dt.JSON | dt.String]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_array_push_json(
    arg: ir.JSONValue | ir.StringValue, value: ir.JSONValue | ir.StringValue
) -> ir.JSONValue | ir.StringValue:
    """
    Append a JSON object to the end of the JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : JSON or string
        JSON value to append

    Returns
    -------
    JSON or string column

    """
    return JSONArrayPushJSON(arg, value).to_expr()


ir.JSONValue.json_array_push_json = json_array_push_json
ir.StringValue.json_array_push_json = json_array_push_json
