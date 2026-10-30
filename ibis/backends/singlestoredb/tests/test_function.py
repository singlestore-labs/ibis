"""Tests for `con.function` with builtin and user-defined functions."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.common.annotations import SignatureValidationError
from ibis.util import gen_name


def _raw(con, *statements):
    with con.begin() as c:
        for sql in statements:
            c.execute(sql)


@pytest.fixture
def blob_vectors(con):
    name = gen_name("blob_vectors")
    _raw(
        con,
        f"CREATE TABLE {name} (g INT, a BLOB, b BLOB)",
        f"INSERT INTO {name} VALUES "
        "(1, JSON_ARRAY_PACK('[1,2,3]'), JSON_ARRAY_PACK('[4,5,6]'))",
    )
    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.fixture
def native_vectors(con):
    name = gen_name("native_vectors")
    _raw(
        con,
        f"CREATE TABLE {name} (g INT, a VECTOR(3), b VECTOR(3))",
        f"INSERT INTO {name} VALUES (1, '[1,2,3]', '[4,5,6]')",
    )
    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.fixture(params=["blob", "native"])
def vectors(request):
    return request.getfixturevalue(f"{request.param}_vectors")


@pytest.fixture
def docs(con):
    name = gen_name("json_docs")
    doc = {
        "name": "Widget",
        "price": 9.5,
        "address": {"city": "Boston", "zip": "02101"},
        "tags": ["a", "b"],
    }
    _raw(
        con,
        f"CREATE TABLE {name} (doc JSON)",
        f"INSERT INTO {name} VALUES ('{json.dumps(doc)}')",
    )
    yield con.table(name)
    con.drop_table(name, force=True)


def _scalar(con, t, expr):
    value = con.execute(t.select(r=expr)).r.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def _as_list(con, t, expr):
    """Execute a vector-valued expression and return a list of numbers."""
    if expr.type().is_binary():
        expr = con.function("json_array_unpack")(expr)
    return [float(x) for x in _scalar(con, t, expr)]


def _is_http(con):
    return "http" in type(con.con).__module__


@pytest.mark.parametrize(
    "other",
    [
        param([1.0, 2.0, 4.0], id="list"),
        param((1, 2, 4), id="tuple"),
        param(np.array([1, 2, 4], dtype="float32"), id="numpy"),
        param("[1, 2, 4]", id="json-string"),
    ],
)
def test_dot_product_literal(con, vectors, other):
    dot_product = con.function("dot_product")
    assert _scalar(con, vectors, dot_product(vectors.a, other)) == 17.0


def test_vector_scalars(con, vectors):
    f = con.function
    a, b = vectors.a, vectors.b
    expr = f("dot_product")(a, b)
    assert expr.type() == dt.float64
    assert _scalar(con, vectors, expr) == 32.0
    assert _scalar(con, vectors, f("euclidean_distance")(a, [1, 2, 4])) == 1.0
    assert _scalar(con, vectors, f("vector_num_elements")(a)) == 3
    assert _scalar(con, vectors, f("vector_elements_sum")(a)) == 6.0
    kth = f("vector_kth_element")(a, 1)
    assert kth.type() == dt.float32
    assert _scalar(con, vectors, kth) == 2.0


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(lambda f, t: f("vector_add")(t.a, t.b), [5, 7, 9], id="add"),
        param(lambda f, t: f("vector_sub")(t.a, [1, 1, 1]), [0, 1, 2], id="sub"),
        param(lambda f, t: f("vector_mul")(t.a, t.b), [4, 10, 18], id="mul"),
        param(lambda f, t: f("scalar_vector_mul")(2, t.a), [2, 4, 6], id="scalar_mul"),
        param(lambda f, t: f("vector_sort")(t.b, "desc"), [6, 5, 4], id="sort"),
    ],
)
def test_vector_valued(con, vectors, make_expr, expected):
    if vectors.a.type().is_array() and _is_http(con):
        pytest.xfail("the HTTP driver returns VECTOR values as raw bytes")
    expr = make_expr(con.function, vectors)
    # vector results keep the input's representation
    assert expr.type() == vectors.a.type()
    assert _as_list(con, vectors, expr) == expected


def test_mixed_vector_kinds_give_vector(con, native_vectors):
    packed = con.function("json_array_pack")("[1, 1, 1]")
    expr = con.function("vector_add")(packed, native_vectors.a)
    assert expr.type() == native_vectors.a.type()
    if not _is_http(con):
        assert _as_list(con, native_vectors, expr) == [2, 3, 4]


def test_vector_sum(con, vectors):
    if vectors.a.type().is_array() and _is_http(con):
        pytest.xfail("the HTTP driver returns VECTOR values as raw bytes")
    vector_sum = con.function("vector_sum")
    expr = vectors.group_by("g").aggregate(r=vector_sum(vectors.a)).r
    unpack = con.function("json_array_unpack")
    if expr.type().is_binary():
        expr = unpack(expr)
    [value] = con.execute(expr)
    assert [float(x) for x in value] == [1, 2, 3]


def test_vector_subvector(con, blob_vectors):
    expr = con.function("vector_subvector")(blob_vectors.a, 1, 2)
    assert _as_list(con, blob_vectors, expr) == [2, 3]


def test_element_type_suffix(con, blob_vectors):
    f = con.function
    left = f("json_array_pack_i8")("[1, 2]")
    expr = f("dot_product_i8")(left, [1, 2])
    assert expr.type() == dt.int64
    assert _scalar(con, blob_vectors, expr) == 5
    kth = f("VECTOR_KTH_ELEMENT_I8")(left, 1)
    assert kth.type() == dt.int8
    assert _scalar(con, blob_vectors, kth) == 2
    packed = f("json_array_pack_f64")("[1.5, 2.5]")
    unpacked = f("json_array_unpack_f64")(packed)
    assert _scalar(con, blob_vectors, unpacked) == [1.5, 2.5]


def test_integer_vector_column(con):
    name = gen_name("int_vectors")
    _raw(
        con,
        f"CREATE TABLE {name} (v VECTOR(3, I8))",
        f"INSERT INTO {name} VALUES ('[1,2,3]')",
    )
    try:
        t = con.table(name)
        # Python sequences are converted to the column's element type
        expr = con.function("dot_product")(t.v, [1, 1, 1])
        assert expr.type() == dt.int64
        assert _scalar(con, t, expr) == 6
        assert con.function("vector_kth_element")(t.v, 0).type() == dt.int8
    finally:
        con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        param("json_get_type", (), "object", id="get_type"),
        param("json_extract_string", ("name",), "Widget", id="extract_string"),
        param("json_extract_string", ("price",), "9.5", id="extract_string_coerces"),
        param("json_extract_string", ("address", "city"), "Boston", id="extract_path"),
        param("json_extract_double", ("price",), 9.5, id="extract_double"),
        param("json_extract_bigint", ("price",), 10, id="extract_bigint_rounds"),
        param("json_extract_bigint", ("name",), 0, id="extract_bigint_coerces"),
        param("json_length", (), 4, id="length"),
        param("json_pretty", (), None, id="pretty"),
    ],
)
def test_json_scalars(con, docs, name, args, expected):
    result = _scalar(con, docs, con.function(name)(docs.doc, *args))
    if expected is None:
        assert json.loads(result)["name"] == "Widget"
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda f, d: f("json_extract_json")(d, "address"),
            {"city": "Boston", "zip": "02101"},
            id="extract_json",
        ),
        param(
            lambda f, d: f("json_keys")(d, "address"), ["city", "zip"], id="keys_path"
        ),
        param(
            lambda f, d: f("json_delete_key")(d, "address", "zip")["address"],
            {"city": "Boston"},
            id="delete_key_path",
        ),
        param(
            lambda f, d: f("json_set_string")(d, "name", "Gadget")["name"],
            "Gadget",
            id="set_string",
        ),
        param(
            lambda f, d: f("json_set_string")(d, "address", "city", "Cambridge")[
                "address"
            ]["city"],
            "Cambridge",
            id="set_string_path",
        ),
        param(
            lambda f, d: f("json_set_double")(d, "price", 1.25)["price"],
            1.25,
            id="set_double",
        ),
        param(
            lambda f, d: f("json_set_json")(d, "tags", '["c"]')["tags"],
            ["c"],
            id="set_json",
        ),
        param(
            lambda f, d: f("json_array_push_string")(d["tags"], "c"),
            ["a", "b", "c"],
            id="push_string",
        ),
        param(
            lambda f, d: f("json_splice_string")(d["tags"], 0, 1, "z"),
            ["z", "b"],
            id="splice_string",
        ),
        param(
            lambda f, d: f("json_include_mask")(d, '{"name": 1}'),
            {"name": "Widget"},
            id="include_mask",
        ),
        param(
            lambda f, d: f("json_exclude_mask")(d, '{"address": 1, "tags": 1}'),
            {"name": "Widget", "price": 9.5},
            id="exclude_mask",
        ),
    ],
)
def test_json_valued(con, docs, make_expr, expected):
    expr = make_expr(con.function, docs.doc)
    assert expr.type().is_json()
    assert _scalar(con, docs, expr) == expected


def test_json_array_contains(con, docs):
    f = con.function("json_array_contains_string")
    assert _scalar(con, docs, f(docs.doc["tags"], "a")) is True
    assert _scalar(con, docs, f(docs.doc["tags"], "z")) is False


@ibis.udf.scalar.builtin(name="HEX")
def _hex(b: bytes) -> str: ...


@ibis.udf.scalar.builtin(name="UNHEX")
def _unhex(s: str) -> bytes: ...


@ibis.udf.scalar.builtin(name="BIT_COUNT")
def _bit_count(x: int) -> int: ...


@ibis.udf.scalar.builtin(name="CONV")
def _conv(s: str, from_base: int, to_base: int) -> str: ...


@ibis.udf.scalar.builtin(name="SIGMOID")
def _sigmoid(x: float) -> float: ...


@ibis.udf.scalar.builtin(name="TRUNCATE")
def _truncate(x: float, decimals: int) -> float: ...


@pytest.mark.parametrize(
    ("fn", "args", "expected"),
    [
        param(_hex, (b"\x01\xff",), "01FF", id="hex"),
        param(_unhex, ("01FF",), b"\x01\xff", id="unhex"),
        param(_bit_count, (7,), 3, id="bit_count"),
        param(_conv, ("ff", 16, 10), "255", id="conv"),
        param(_sigmoid, (0.0,), 0.5, id="sigmoid"),
        param(_truncate, (1.987, 1), 1.9, id="truncate"),
    ],
)
def test_builtin_udf(con, docs, fn, args, expected):
    # Builtins that need no special handling are declared as builtin UDFs
    # rather than looked up with con.function
    assert _scalar(con, docs, fn(*args)) == expected


def test_simple_builtin_is_not_a_function(con):
    with pytest.raises(com.MissingUDFError):
        con.function("sigmoid")


@pytest.mark.parametrize(
    ("name", "args", "match"),
    [
        param("dot_product", (1,), "takes 2 arguments, but 1", id="too_few"),
        param("vector_sort", (1, 2, 3), "takes 1 to 2 arguments", id="too_many"),
        param("json_set_string", (1, 2), "takes at least 3", id="variadic"),
    ],
)
def test_builtin_arity(name, args, match):
    from ibis.backends.singlestoredb import functions

    with pytest.raises(TypeError, match=match):
        functions.lookup(name)(*args)


def test_builtin_compiles_unquoted():
    from ibis.backends.singlestoredb import functions

    t = ibis.table({"v": "array<float32, 3>", "doc": "json"}, name="t")
    expr = t.select(
        a=functions.lookup("vector_sort")(t.v, "desc"),
        b=functions.lookup("json_extract_string")(t.doc, "a", "b"),
    )
    sql = ibis.to_sql(expr, dialect="singlestoredb")
    assert "VECTOR_SORT(`t0`.`v`, 'desc')" in sql
    assert "JSON_EXTRACT_STRING(`t0`.`doc`, 'a', 'b')" in sql


def test_unknown_name_is_not_a_builtin():
    from ibis.backends.singlestoredb import functions

    assert functions.lookup("no_such_function") is None
    assert functions.lookup("json_get_type_i8") is None


@pytest.fixture
def other_database(con):
    name = gen_name("udf_db")
    con.create_database(name)
    yield name
    con.drop_database(name, force=True)


@pytest.fixture
def routines(con, other_database):
    names = {
        key: gen_name(key)
        for key in ["add", "vec", "proc", "tvf", "init", "iter", "merge", "term", "agg"]
    }
    n = names
    _raw(
        con,
        f"CREATE FUNCTION {n['add']}(a INT, `from` BIGINT) RETURNS BIGINT "
        "AS BEGIN RETURN a + `from`; END",
        f"CREATE FUNCTION {n['vec']}(v VECTOR(3), s TEXT) RETURNS DOUBLE "
        "AS BEGIN RETURN DOT_PRODUCT(v, v) + LENGTH(s); END",
        f"CREATE PROCEDURE {n['proc']}(x INT) AS BEGIN ECHO SELECT x; END",
        f"CREATE FUNCTION {n['tvf']}(x INT) RETURNS TABLE AS RETURN SELECT x AS y",
        f"CREATE FUNCTION {n['init']}() RETURNS BIGINT AS BEGIN RETURN 0; END",
        f"CREATE FUNCTION {n['iter']}(s BIGINT, v BIGINT) RETURNS BIGINT "
        "AS BEGIN RETURN s + v; END",
        f"CREATE FUNCTION {n['merge']}(a BIGINT, b BIGINT) RETURNS BIGINT "
        "AS BEGIN RETURN a + b; END",
        f"CREATE FUNCTION {n['term']}(s BIGINT) RETURNS BIGINT AS BEGIN RETURN s; END",
        f"CREATE AGGREGATE {n['agg']}(BIGINT) RETURNS BIGINT WITH STATE BIGINT "
        f"INITIALIZE WITH {n['init']} ITERATE WITH {n['iter']} "
        f"MERGE WITH {n['merge']} TERMINATE WITH {n['term']}",
        f"CREATE FUNCTION {other_database}.{n['add']}(a INT, b INT) RETURNS INT "
        "AS BEGIN RETURN a * b; END",
    )
    yield names
    _raw(
        con,
        f"DROP AGGREGATE IF EXISTS {n['agg']}",
        f"DROP PROCEDURE IF EXISTS {n['proc']}",
        *(
            f"DROP FUNCTION IF EXISTS {n[key]}"
            for key in ["add", "vec", "tvf", "init", "iter", "merge", "term"]
        ),
    )


def test_udf(con, native_vectors, routines):
    my_add = con.function(routines["add"])
    # `from` is not a valid Python parameter name
    assert list(my_add.__signature__.parameters) == ["a", "arg1"]
    expr = my_add(native_vectors.g, 41)
    assert expr.type() == dt.int64
    assert _scalar(con, native_vectors, expr) == 42

    # names are case-insensitive, and parameters are typed
    my_vec = con.function(routines["vec"].upper())
    expr = my_vec(native_vectors.a, "ab")
    assert expr.type() == dt.float64
    assert _scalar(con, native_vectors, expr) == 16.0
    with pytest.raises(SignatureValidationError):
        my_vec(native_vectors.g, "ab")


def test_udf_in_other_database(con, native_vectors, routines, other_database):
    my_mul = con.function(routines["add"], database=other_database)
    assert _scalar(con, native_vectors, my_mul(native_vectors.g, 41)) == 41


def test_udaf(con, native_vectors, routines):
    my_sum = con.function(routines["agg"])
    t = native_vectors
    expr = t.group_by("g").aggregate(total=my_sum(t.g), none=my_sum(t.g, where=t.g > 1))
    assert expr.total.type() == dt.int64
    result = con.execute(expr)
    assert result.total.iloc[0] == 1
    # like builtin aggregates, NULL when no rows match
    assert result.none.isna().all()


@pytest.mark.parametrize(
    ("key", "exception", "match"),
    [
        param("proc", com.UnsupportedOperationError, "stored procedure", id="proc"),
        param("tvf", com.UnsupportedOperationError, "table-valued", id="tvf"),
        param("missing", com.MissingUDFError, "", id="missing"),
    ],
)
def test_udf_errors(con, routines, key, exception, match):
    with pytest.raises(exception, match=match):
        con.function(routines.get(key, "no_such_function"))


def test_builtins_are_not_looked_up_in_a_database(con, other_database):
    with pytest.raises(com.MissingUDFError):
        con.function("dot_product", database=other_database)


def test_builtin_attribute_access(con):
    # builtins are cached, and attributes are the same as calls
    assert con.function.dot_product is con.function("DOT_PRODUCT")
    assert con.function.dot_product_i8 is con.function("dot_product_i8")
    assert {"dot_product", "json_get_type", "vector_sum"} <= set(dir(con.function))


def test_attribute_errors(con):
    assert not hasattr(con.function, "no_such_function")
    assert not hasattr(con.function, "_repr_html_")
    with pytest.raises(AttributeError, match="no_such_function"):
        con.function.no_such_function  # noqa: B018


def test_udf_attribute_access_is_not_cached(con, native_vectors, routines):
    name = routines["term"]
    my_term = getattr(con.function, name)
    assert my_term(native_vectors.g).type() == dt.int64

    _raw(
        con,
        f"CREATE OR REPLACE FUNCTION {name}(s BIGINT) RETURNS DOUBLE "
        "AS BEGIN RETURN s / 2; END",
    )
    expr = getattr(con.function, name)(native_vectors.g)
    assert expr.type() == dt.float64
    assert _scalar(con, native_vectors, expr) == 0.5
