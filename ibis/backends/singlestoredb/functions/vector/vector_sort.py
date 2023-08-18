from typing import Optional

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class VectorSort(Value):
    """32-bit float vector sort."""

    arg = Value[dt.Binary]
    direction = dt.String

    dtype = dt.binary
    shape = rlz.shape_like('arg')


def vector_sort(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSort(arg, direction).to_expr()


ir.BinaryValue.vector_sort = vector_sort


class VectorSortI8(VectorSort):
    """8-bit integer vector sort."""


def vector_sort_i8(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in an 8-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortI8(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i8 = vector_sort_i8


class VectorSortI16(VectorSort):
    """16-bit integer vector sort."""


def vector_sort_i16(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 16-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortI16(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i16 = vector_sort_i16


class VectorSortI32(VectorSort):
    """32-bit integer vector sort."""


def vector_sort_i32(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortI32(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i32 = vector_sort_i32


class VectorSortI64(VectorSort):
    """64-bit integer vector sort."""


def vector_sort_i64(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 64-bit int vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortI64(arg, direction).to_expr()


ir.BinaryValue.vector_sort_i64 = vector_sort_i64


class VectorSortF32(VectorSort):
    """32-bit float vector sort."""


def vector_sort_f32(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 32-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortF32(arg, direction).to_expr()


ir.BinaryValue.vector_sort_f32 = vector_sort_f32


class VectorSortF64(VectorSort):
    """64-bit float vector sort."""


def vector_sort_f64(
    arg: ir.BinaryValue, direction: Optional[ir.StringValue] = 'asc'
) -> ir.BinaryValue:
    """
    Sort the elements in a 64-bit float vector.

    Parameters
    ----------
    arg : blob
        Vector expression
    direction : str, optional
        Direction of the sort: 'asc' or 'desc'

    Returns
    -------
    Blob column

    """
    return VectorSortF64(arg, direction).to_expr()


ir.BinaryValue.vector_sort_f64 = vector_sort_f64
