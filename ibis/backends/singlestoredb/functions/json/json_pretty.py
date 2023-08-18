import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class JSONPretty(Value):
    """Pretty-print JSON."""

    arg = Value[dt.JSON | dt.String]

    dtype = dt.string
    shape = rlz.shape_like('arg')


def json_pretty(arg: ir.JSONValue | ir.StringValue) -> ir.StringValue:
    """
    Pretty-print a JSON object or array.

    Parameters
    ----------
    arg : JSON or string
        JSON array

    Returns
    -------
    String column

    """
    return JSONPretty(arg).to_expr()


ir.JSONValue.json_pretty = json_pretty
ir.StringValue.json_pretty = json_pretty
