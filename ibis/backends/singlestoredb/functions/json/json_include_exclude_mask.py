import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class JSONExcludeMask(Value):
    """JSON exclude using a mask."""

    arg = Value[dt.JSON | dt.String]
    mask = dt.String

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_exclude_mask(
    arg: ir.JSONValue | ir.StringValue, mask: ir.JSONValue | ir.StringValue
) -> ir.JSONValue | ir.StringValue:
    """
    Get a subset of a JSON object after applying a mask.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    mask : JSON or string
        Mask to apply

    Returns
    -------
    JSON or String column

    """
    return JSONExcludeMask(arg, mask).to_expr()


ir.JSONValue.json_exclude_mask = json_exclude_mask
ir.StringValue.json_exclude_mask = json_exclude_mask


class JSONIncludeMask(Value):
    """JSON include using a mask."""

    arg = Value[dt.JSON | dt.String]
    mask = dt.String

    dtype = rlz.dtype_like('arg')
    shape = rlz.shape_like('arg')


def json_include_mask(
    arg: ir.JSONValue | ir.StringValue, mask: ir.JSONValue | ir.StringValue
) -> ir.JSONValue | ir.StringValue:
    """
    Get a subset of a JSON object after applying a mask.

    Parameters
    ----------
    arg : JSON or string
        JSON array
    mask : JSON or string
        Mask to apply

    Returns
    -------
    JSON or String column

    """
    return JSONIncludeMask(arg, mask).to_expr()


ir.JSONValue.json_include_mask = json_include_mask
ir.StringValue.json_include_mask = json_include_mask
