import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class VectorSub(Value):
    """32-bit float vector subtraction."""

    left = Value[dt.Binary]
    right = Value[dt.Binary]

    dtype = dt.binary
    shape = rlz.shape_like('left')


def vector_sub(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 32-bit float vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSub(left, right).to_expr()


ir.BinaryValue.vector_sub = vector_sub


class VectorSubI8(VectorSub):
    """8-bit integer vector subtraction."""


def vector_sub_i8(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 8-bit int vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubI8(left, right).to_expr()


ir.BinaryValue.vector_sub_i8 = vector_sub_i8


class VectorSubI16(VectorSub):
    """16-bit integer vector subtraction."""


def vector_sub_i16(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 16-bit int vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubI16(left, right).to_expr()


ir.BinaryValue.vector_sub_i16 = vector_sub_i16


class VectorSubI32(VectorSub):
    """32-bit integer vector subtraction."""


def vector_sub_i32(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 32-bit int vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubI32(left, right).to_expr()


ir.BinaryValue.vector_sub_i32 = vector_sub_i32


class VectorSubI64(VectorSub):
    """64-bit integer vector subtraction."""


def vector_sub_i64(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 64-bit int vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubI64(left, right).to_expr()


ir.BinaryValue.vector_sub_i64 = vector_sub_i64


class VectorSubF32(VectorSub):
    """32-bit float vector subtraction."""


def vector_sub_f32(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 32-bit float vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubF32(left, right).to_expr()


ir.BinaryValue.vector_sub_f32 = vector_sub_f32


class VectorSubF64(VectorSub):
    """64-bit float vector subtraction."""


def vector_sub_f64(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Subtract the second 64-bit float vector from the first.

    Parameters
    ----------
    left : blob
        Vector expression
    right : blob
        Vector expression

    Returns
    -------
    Blob column

    """
    return VectorSubF64(left, right).to_expr()


ir.BinaryValue.vector_sub_f64 = vector_sub_f64
