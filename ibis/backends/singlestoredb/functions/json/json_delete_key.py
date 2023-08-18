import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value
from ibis.common.typing import VarTuple


class JSONDeleteKey(Value):
    """JSON delete key."""

    arg = Value[dt.JSON | dt.String]
    # TODO: min_length=1
    key_path = VarTuple[Value[dt.String | dt.Integer]]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_delete_key(
    arg: ir.JSONValue | ir.StringValue,
    *key_path: ir.StringValue | ir.NumericValue,
) -> ir.JSONValue | ir.StringValue:
    """
    Delete a key from a JSON map or array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Keys or zero-indexed array positions.

    Returns
    -------
    JSON or string column

    """
    return JSONDeleteKey(arg, key_path).to_expr()


ir.JSONValue.json_delete_key = json_delete_key
ir.StringValue.json_delete_key = json_delete_key
