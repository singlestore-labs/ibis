import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value

StrictNumeric = Value[dt.Numeric]


class JSONArrayContainsDouble(Value):
    """JSON array element test."""

    arg = Value[dt.JSON | dt.String]
    value = StrictNumeric

    dtype = dt.bool
    shape = rlz.shape_like('arg')


def json_array_contains_double(
    arg: ir.JSONValue | ir.StringValue, value: ir.NumericValue
) -> ir.BooleanValue:
    """
    Does the array contain the given float value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : float
        Float value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsDouble(arg, value).to_expr()


ir.JSONValue.json_array_contains_double = json_array_contains_double
ir.StringValue.json_array_contains_double = json_array_contains_double


class JSONArrayContainsString(Value):
    """JSON array element test."""

    arg = Value[dt.JSON | dt.String]
    value = Value[dt.String]

    dtype = dt.bool
    shape = rlz.shape_like('arg')


def json_array_contains_string(
    arg: ir.JSONValue | ir.StringValue, value: ir.StringValue
) -> ir.BooleanValue:
    """
    Does the array contain the given string value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        String value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsString(arg, value).to_expr()


ir.JSONValue.json_array_contains_string = json_array_contains_string
ir.StringValue.json_array_contains_string = json_array_contains_string


class JSONArrayContainsJSON(Value):
    """JSON array element test."""

    arg = Value[dt.JSON | dt.String]
    value = Value[dt.String]

    dtype = dt.bool
    shape = rlz.shape_like('arg')


def json_array_contains_json(
    arg: ir.JSONValue | ir.StringValue, value: ir.StringValue
) -> ir.BooleanValue:
    """
    Does the array contain the given JSON value?

    Parameters
    ----------
    arg : JSON or string
        JSON array
    value : string
        JSON value to search for

    Returns
    -------
    Boolean column

    """
    return JSONArrayContainsJSON(arg, value).to_expr()


ir.JSONValue.json_array_contains_json = json_array_contains_json
ir.StringValue.json_array_contains_json = json_array_contains_json
