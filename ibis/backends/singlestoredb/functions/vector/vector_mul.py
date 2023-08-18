import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.expr.operations import Value


class VectorMul(Value):
    """32-bit float vector multiplication."""

    left = Value[dt.Binary]
    right = Value[dt.Binary]

    dtype = dt.binary
    shape = rlz.shape_like('left')


def vector_mul(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 32-bit float vector blobs.

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
    return VectorMul(left, right).to_expr()


ir.BinaryValue.vector_mul = vector_mul


class VectorMulI8(VectorMul):
    """8-bit integer vector multiplication."""


def vector_mul_i8(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 8-bit int vector blobs.

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
    return VectorMulI8(left, right).to_expr()


ir.BinaryValue.vector_mul_i8 = vector_mul_i8


class VectorMulI16(VectorMul):
    """16-bit integer vector multiplication."""


def vector_mul_i16(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 16-bit int vector blobs.

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
    return VectorMulI16(left, right).to_expr()


ir.BinaryValue.vector_mul_i16 = vector_mul_i16


class VectorMulI32(VectorMul):
    """32-bit integer vector multiplication."""


def vector_mul_i32(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 32-bit int vector blobs.

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
    return VectorMulI32(left, right).to_expr()


ir.BinaryValue.vector_mul_i32 = vector_mul_i32


class VectorMulI64(VectorMul):
    """64-bit integer vector multiplication."""


def vector_mul_i64(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 64-bit int vector blobs.

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
    return VectorMulI64(left, right).to_expr()


ir.BinaryValue.vector_mul_i64 = vector_mul_i64


class VectorMulF32(VectorMul):
    """32-bit float vector multiplication."""


def vector_mul_f32(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 32-bit float vector blobs.

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
    return VectorMulF32(left, right).to_expr()


ir.BinaryValue.vector_mul_f32 = vector_mul_f32


class VectorMulF64(VectorMul):
    """64-bit float vector multiplication."""


def vector_mul_f64(left: ir.BinaryValue, right: ir.BinaryValue) -> ir.BinaryValue:
    """
    Multiplies two 64-bit float vector blobs.

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
    return VectorMulF64(left, right).to_expr()


ir.BinaryValue.vector_mul_f64 = vector_mul_f64
