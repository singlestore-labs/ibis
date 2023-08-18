import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class JSONGetType(Value):
    """JSON type getter."""

    arg = Value[dt.JSON | dt.String]

    dtype = dt.string
    shape = rlz.shape_like('arg')


def json_get_type(arg: ir.JSONValue | ir.StringValue) -> ir.StringValue:
    """
    Get the type of the JSON object.

    Parameters
    ----------
    arg : JSON or string
        JSON array

    Returns
    -------
    String column

    """
    return JSONGetType(arg).to_expr()


ir.JSONValue.json_get_type = json_get_type
ir.StringValue.json_get_type = json_get_type
