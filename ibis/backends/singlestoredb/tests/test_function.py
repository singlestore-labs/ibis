"""Tests for the SingleStoreDB `functions` package and `con.function`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.singlestoredb import functions as s2
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
        expr = s2.vector.to_json(expr)
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
    assert _scalar(con, vectors, s2.vector.dot_product(vectors.a, other)) == 17.0


def test_vector_scalars(con, vectors):
    a, b = vectors.a, vectors.b
    expr = s2.vector.dot_product(a, b)
    assert expr.type() == dt.float64
    assert _scalar(con, vectors, expr) == 32.0
    assert _scalar(con, vectors, s2.vector.euclidean_distance(a, [1, 2, 4])) == 1.0
    assert _scalar(con, vectors, s2.vector.num_elements(a)) == 3
    assert _scalar(con, vectors, s2.vector.elements_sum(a)) == 6.0
    kth = s2.vector.kth_element(a, 1)
    assert kth.type() == dt.float32
    assert _scalar(con, vectors, kth) == 2.0


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(lambda t: s2.vector.add(t.a, t.b), [5, 7, 9], id="add"),
        param(lambda t: s2.vector.sub(t.a, [1, 1, 1]), [0, 1, 2], id="sub"),
        param(lambda t: s2.vector.mul(t.a, t.b), [4, 10, 18], id="mul"),
        param(lambda t: s2.vector.scalar_mul(2, t.a), [2, 4, 6], id="scalar_mul"),
        param(lambda t: s2.vector.sort(t.b), [4, 5, 6], id="sort"),
        param(lambda t: s2.vector.sort(t.b, "desc"), [6, 5, 4], id="sort_desc"),
    ],
)
def test_vector_valued(con, vectors, make_expr, expected):
    if vectors.a.type().is_array() and _is_http(con):
        pytest.xfail("the HTTP driver returns VECTOR values as raw bytes")
    expr = make_expr(vectors)
    # vector results keep the input's representation
    assert expr.type() == vectors.a.type()
    assert _as_list(con, vectors, expr) == expected


def test_mixed_vector_kinds_give_vector(con, native_vectors):
    packed = s2.vector.from_json("[1, 1, 1]")
    expr = s2.vector.add(packed, native_vectors.a)
    assert expr.type() == native_vectors.a.type()
    if not _is_http(con):
        assert _as_list(con, native_vectors, expr) == [2, 3, 4]


def test_vector_sum(con, vectors):
    if vectors.a.type().is_array() and _is_http(con):
        pytest.xfail("the HTTP driver returns VECTOR values as raw bytes")
    expr = vectors.group_by("g").aggregate(r=s2.vector.sum(vectors.a)).r
    if expr.type().is_binary():
        expr = s2.vector.to_json(expr)
    [value] = con.execute(expr)
    assert [float(x) for x in value] == [1, 2, 3]


def test_vector_subvector(con, blob_vectors):
    expr = s2.vector.subvector(blob_vectors.a, 1, 2)
    assert _as_list(con, blob_vectors, expr) == [2, 3]


def test_blob_element_type(con, blob_vectors):
    # packed vectors don't record their element type
    left = s2.vector.from_json("[1, 2]", element_type="i8")
    expr = s2.vector.dot_product(left, [1, 2], element_type="i8")
    assert expr.type() == dt.int64
    assert _scalar(con, blob_vectors, expr) == 5
    kth = s2.vector.kth_element(left, 1, element_type="i8")
    assert kth.type() == dt.int8
    assert _scalar(con, blob_vectors, kth) == 2
    packed = s2.vector.from_json("[1.5, 2.5]", element_type="f64")
    unpacked = s2.vector.to_json(packed, element_type="f64")
    assert _scalar(con, blob_vectors, unpacked) == [1.5, 2.5]


@pytest.fixture
def int_vectors(con):
    name = gen_name("int_vectors")
    _raw(
        con,
        f"CREATE TABLE {name} (v VECTOR(3, I8), f VECTOR(3), j JSON)",
        f"INSERT INTO {name} VALUES ('[1,2,3]', '[1,2,3]', '[1,1,1]')",
    )
    yield con.table(name)
    con.drop_table(name, force=True)


def test_element_type_from_vector(con, int_vectors):
    t = int_vectors
    # Python sequences and JSON arrays are converted to the VECTOR's type
    for other in ([1, 1, 1], t.j.array):
        expr = s2.vector.dot_product(t.v, other)
        assert expr.type() == dt.int64
        assert _scalar(con, t, expr) == 6
    assert s2.vector.kth_element(t.v, 0).type() == dt.int8
    assert "DOT_PRODUCT(" in ibis.to_sql(s2.vector.dot_product(t.v, [1, 1, 1]))


def test_element_type_from_json_arrays(con, int_vectors):
    t = int_vectors
    # without a VECTOR, JSON arrays are packed as 32-bit floats
    expr = s2.vector.dot_product(t.j.array, [1, 2, 3])
    assert expr.type() == dt.float64
    assert _scalar(con, t, expr) == 6.0
    assert _scalar(con, t, s2.vector.elements_sum(t.j.array)) == 3.0


def test_element_type_from_array_type():
    t = ibis.table({"a": "array<int8>"}, name="t")
    expr = s2.vector.dot_product(t.a, [1, 2])
    assert expr.type() == dt.int64
    sql = ibis.to_sql(expr, dialect="singlestoredb")
    assert "DOT_PRODUCT_I8(JSON_ARRAY_PACK_I8(`t0`.`a`), JSON_ARRAY_PACK_I8(" in sql


@pytest.mark.parametrize(
    ("make_expr", "exception", "match"),
    [
        param(
            lambda t: s2.vector.dot_product(t.v, [1, 1, 1], element_type="f32"),
            ValueError,
            "doesn't match",
            id="mismatch",
        ),
        param(
            lambda t: s2.vector.dot_product(t.v, t.v, element_type="u8"),
            ValueError,
            "must be one of",
            id="unknown",
        ),
        param(
            lambda t: s2.vector.from_json(t.v), TypeError, "VECTOR", id="pack_vector"
        ),
    ],
)
def test_element_type_errors(make_expr, exception, match):
    t = ibis.table({"v": "array<!int8, 3>"}, name="t")
    with pytest.raises(exception, match=match):
        make_expr(t)


def test_mismatched_vectors_raise(con, int_vectors):
    t = int_vectors
    with pytest.raises(Exception, match="incompatib"):
        _scalar(con, t, s2.vector.dot_product(t.v, t.f))


@pytest.mark.parametrize(
    ("fn", "args", "expected"),
    [
        param(s2.json.get_type, (), "object", id="get_type"),
        param(s2.json.extract_string, ("name",), "Widget", id="extract_string"),
        param(s2.json.extract_string, ("price",), "9.5", id="extract_string_coerces"),
        param(s2.json.extract_string, ("address", "city"), "Boston", id="extract_path"),
        param(s2.json.extract_double, ("price",), 9.5, id="extract_double"),
        param(s2.json.extract_bigint, ("price",), 10, id="extract_bigint_rounds"),
        param(s2.json.extract_bigint, ("name",), 0, id="extract_bigint_coerces"),
        param(s2.json.length, (), 4, id="length"),
        param(s2.json.match_any_exists, ("address", "zip"), True, id="exists"),
        param(s2.json.match_any_exists, ("address", "x"), False, id="not_exists"),
        param(s2.json.pretty, (), None, id="pretty"),
    ],
)
def test_json_scalars(con, docs, fn, args, expected):
    result = _scalar(con, docs, fn(docs.doc, *args))
    if expected is None:
        assert json.loads(result)["name"] == "Widget"
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda d: s2.json.extract_json(d, "address"),
            {"city": "Boston", "zip": "02101"},
            id="extract_json",
        ),
        param(lambda d: s2.json.keys(d, "address"), ["city", "zip"], id="keys_path"),
        param(
            lambda d: s2.json.delete_key(d, "address", "zip")["address"],
            {"city": "Boston"},
            id="delete_key_path",
        ),
        param(
            lambda d: s2.json.set_string(d, "name", "Gadget")["name"],
            "Gadget",
            id="set_string",
        ),
        param(
            lambda d: s2.json.set_string(d, "address", "city", "Cambridge")["address"][
                "city"
            ],
            "Cambridge",
            id="set_string_path",
        ),
        param(
            lambda d: s2.json.set_double(d, "price", 1.25)["price"],
            1.25,
            id="set_double",
        ),
        param(
            lambda d: s2.json.set_json(d, "tags", '["c"]')["tags"], ["c"], id="set_json"
        ),
        param(
            lambda d: s2.json.array_push_string(d["tags"], "c"),
            ["a", "b", "c"],
            id="push_string",
        ),
        param(
            lambda d: s2.json.array_push_double(d["tags"], 1.5),
            ["a", "b", 1.5],
            id="push_double",
        ),
        param(
            lambda d: s2.json.array_push_json(d["tags"], '{"x": 1}'),
            ["a", "b", {"x": 1}],
            id="push_json",
        ),
        param(
            lambda d: s2.json.splice_string(d["tags"], 0, 1, "z"),
            ["z", "b"],
            id="splice_string",
        ),
        param(
            lambda d: s2.json.splice_double(d["tags"], 1, 1, 7, 8),
            ["a", 7, 8],
            id="splice_double",
        ),
        param(
            lambda d: s2.json.splice_json(d["tags"], 2, 0, "[1]"),
            ["a", "b", [1]],
            id="splice_json",
        ),
        param(
            lambda d: s2.json.include_mask(d, '{"name": 1}'),
            {"name": "Widget"},
            id="include_mask",
        ),
        param(
            lambda d: s2.json.exclude_mask(d, '{"address": 1, "tags": 1}'),
            {"name": "Widget", "price": 9.5},
            id="exclude_mask",
        ),
        param(
            lambda d: s2.json.merge_patch(d["address"], '{"zip": null, "st": "MA"}'),
            {"city": "Boston", "st": "MA"},
            id="merge_patch",
        ),
        param(
            lambda d: s2.json.build_array(1, "a", d["price"]),
            [1, "a", 9.5],
            id="build_array",
        ),
        param(
            lambda d: s2.json.build_object({"n": d["name"], "k": 1}),
            {"n": "Widget", "k": 1},
            id="build_object_mapping",
        ),
        param(
            lambda d: s2.json.build_object("n", d["name"], "k", 1),
            {"n": "Widget", "k": 1},
            id="build_object_pairs",
        ),
        param(lambda d: s2.json.from_value(d["price"]), 9.5, id="from_value"),
    ],
)
def test_json_valued(con, docs, make_expr, expected):
    expr = make_expr(docs.doc)
    assert expr.type().is_json()
    assert _scalar(con, docs, expr) == expected


@pytest.mark.parametrize(
    "fn",
    [
        s2.json.array_contains_string,
        s2.json.array_contains_double,
        s2.json.array_contains_json,
    ],
)
def test_json_array_contains(con, docs, fn):
    found, missing = {
        s2.json.array_contains_string: ("a", "z"),
        s2.json.array_contains_double: (1.5, 2.5),
        s2.json.array_contains_json: ('"a"', '"z"'),
    }[fn]
    array = s2.json.array_push_double(docs.doc["tags"], 1.5)
    assert _scalar(con, docs, fn(array, found)) is True
    assert _scalar(con, docs, fn(array, missing)) is False


@pytest.fixture
def scores(con):
    name = gen_name("scores")
    _raw(
        con,
        f"CREATE TABLE {name} (id INT, doc JSON)",
        f"""INSERT INTO {name} VALUES (1, '{{"s": [1, 5, 9], "n": ["a", "b"]}}'),
        (2, '{{"s": [2], "n": []}}')""",
    )
    yield con.table(name)
    con.drop_table(name, force=True)


def _column(con, t, expr):
    return con.execute(t.select("id", r=expr).order_by("id")).r.tolist()


def test_json_match_any(con, scores):
    d = scores.doc
    even = s2.json.match_any(lambda v: v % 2 == 0, d, "s", value_type="bigint")
    assert _column(con, scores, even) == [False, True]
    big = s2.json.match_any(lambda v: v > 4, d, "s", value_type="double")
    assert _column(con, scores, big) == [True, False]
    named = s2.json.match_any(lambda v: v == "b", d, "n", value_type="string")
    assert _column(con, scores, named) == [True, False]
    with pytest.raises(ValueError, match="value_type"):
        s2.json.match_any(lambda v: v, d, "s", value_type="int")


def test_json_agg(con, scores):
    [result] = con.execute(scores.aggregate(r=s2.json.agg(scores.id)).r)
    assert sorted(result) == [1, 2]
    [result] = con.execute(
        scores.filter(scores.id > 1).aggregate(r=s2.json.agg(scores.doc["s"])).r
    )
    assert result == [[2]]


def test_reduce(con, scores):
    d = scores.doc
    total = s2.json.reduce(0, d["s"], lambda acc, v: acc + v, value_type="bigint")
    assert total.type().is_integer()
    assert _column(con, scores, total) == [15, 2]
    joined = s2.json.reduce(
        "", d["n"], lambda acc, v: acc.concat(v), value_type="string"
    )
    assert _column(con, scores, joined) == ["ab", ""]
    indexes = s2.json.reduce(0, d["s"], lambda acc, _, i: acc + i)
    assert _column(con, scores, indexes) == [3, 0]


@pytest.mark.parametrize(
    ("make_expr", "match"),
    [
        param(s2.json.build_object, "alternating", id="build_object"),
        param(lambda d: s2.json.set_string(d, "a"), "at least one key", id="set"),
        param(s2.json.delete_key, "at least one key", id="delete"),
        param(lambda d: s2.json.extract_string(d, 1.5), "keys must be", id="key_type"),
        param(
            lambda d: s2.json.reduce(0, d, lambda a, _: a, value_type="x"),
            "value_type",
            id="reduce",
        ),
    ],
)
def test_json_argument_errors(make_expr, match):
    d = ibis.table({"doc": "json"}, name="t").doc
    with pytest.raises((TypeError, ValueError), match=match):
        make_expr(d)


def test_functions_compile_unquoted():
    t = ibis.table({"v": "array<float32, 3>", "doc": "json"}, name="t")
    expr = t.select(
        a=s2.vector.sort(t.v, "desc"), b=s2.json.extract_string(t.doc, "a", "b")
    )
    sql = ibis.to_sql(expr, dialect="singlestoredb")
    assert "VECTOR_SORT(`t0`.`v`, 'desc')" in sql
    assert "JSON_EXTRACT_STRING(`t0`.`doc`, 'a', 'b')" in sql


def test_submodules():
    # functions are only available from their module
    assert s2.__all__ == ["json", "vector"]
    assert not hasattr(s2, "dot_product")
    for module in (s2.json, s2.vector):
        assert all(callable(getattr(module, name)) for name in module.__all__)
        assert not any(name.startswith(("json_", "vector_")) for name in module.__all__)
        for name in module.__all__:
            assert "Calls `" in getattr(module, name).__doc__


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


@pytest.mark.parametrize("name", ["sigmoid", "dot_product", "json_keys"])
def test_builtins_are_not_functions(con, name):
    # con.function only finds functions defined on the server
    with pytest.raises(com.MissingUDFError):
        con.function(name)


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


def test_udf_is_looked_up_on_each_call(con, native_vectors, routines):
    name = routines["term"]
    my_term = con.function(name)
    assert my_term(native_vectors.g).type() == dt.int64

    _raw(
        con,
        f"CREATE OR REPLACE FUNCTION {name}(s BIGINT) RETURNS DOUBLE "
        "AS BEGIN RETURN s / 2; END",
    )
    expr = con.function(name)(native_vectors.g)
    assert expr.type() == dt.float64
    assert _scalar(con, native_vectors, expr) == 0.5
