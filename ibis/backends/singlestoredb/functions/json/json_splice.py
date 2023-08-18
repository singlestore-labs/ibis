import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value
from ibis.common.typing import VarTuple

StrictNumeric = Value[dt.Numeric]


class JSONSpliceDouble(Value):
    """JSON splice double values."""

    arg = Value[dt.JSON | dt.String]
    start = dt.Integer
    length = dt.Integer
    # TODO: min_length=1
    values = VarTuple[StrictNumeric]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_splice_double(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.FloatingValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice double values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : doubles
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceDouble(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_double = json_splice_double
ir.StringValue.json_splice_double = json_splice_double


class JSONSpliceString(Value):
    """JSON splice string values."""

    arg = Value[dt.JSON | dt.String]
    start = dt.Integer
    length = dt.Integer
    # TODO: min_length=1
    values = VarTuple[dt.String]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_splice_string(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice double values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : strings
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceString(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_string = json_splice_string
ir.StringValue.json_splice_string = json_splice_string


class JSONSpliceJSON(Value):
    """JSON splice JSON values."""

    arg = Value[dt.JSON | dt.String]
    start = dt.Integer
    length = dt.Integer
    # TODO: min_length=1
    values = VarTuple[Value[dt.String | dt.JSON]]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_splice_json(
    arg: ir.JSONValue | ir.StringValue,
    start: ir.IntegerValue,
    length: ir.IntegerValue,
    *values: ir.JSONValue | ir.StringValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Splice JSON values at the specified position.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    start : int
        Zero-indexed position to start splice
    length : int
        Length of the splice
    *values : strings
        Values to insert

    Returns
    -------
    String or JSON Column

    """
    return JSONSpliceJSON(arg, start, length, values).to_expr()


ir.JSONValue.json_splice_json = json_splice_json
ir.StringValue.json_splice_json = json_splice_json
