import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value
from ibis.common.typing import VarTuple


class JSONKeys(Value):
    """JSON exclude using a mask."""

    arg = Value[dt.JSON | dt.String]
    key_path = VarTuple[Value[dt.String | dt.Integer]]

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_keys(
    arg: ir.JSONValue | ir.StringValue, *key_path: ir.StringValue | ir.NumericValue
) -> ir.JSONValue | ir.StringValue:
    """
    Return the top-level keys of a JSON object in the form of a JSON array.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    *key_path : strings or ints
        Path to a JSON object

    Returns
    -------
    JSON or String column

    """
    return JSONKeys(arg, key_path).to_expr()


ir.JSONValue.json_keys = json_keys
ir.StringValue.json_keys = json_keys
