import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class JSONLength(Value):
    """JSON object / array length."""

    arg = Value[dt.JSON | dt.String]

    dtype = dt.int
    shape = rlz.shape_like('arg')


def json_length(arg: ir.JSONValue | ir.StringValue) -> ir.IntegerValue:
    """
    Get the length of the JSON object / array.

    Parameters
    ----------
    arg : JSON or string
        JSON array

    Returns
    -------
    Integer column

    """
    return JSONLength(arg).to_expr()


ir.JSONValue.json_length = json_length
ir.StringValue.json_length = json_length
