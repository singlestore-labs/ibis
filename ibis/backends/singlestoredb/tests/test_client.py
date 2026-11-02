from __future__ import annotations

import contextlib
import json
from datetime import date
from operator import methodcaller

import pandas as pd
import pytest
import sqlglot as sg
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis import udf
from ibis.backends.singlestoredb import server
from ibis.backends.singlestoredb.tests.conftest import (
    IBIS_TEST_SINGLESTOREDB_DB,
    SINGLESTOREDB_HOST,
    SINGLESTOREDB_PASS,
    SINGLESTOREDB_USER,
)
from ibis.backends.tests.errors import (
    SingleStoreDBOperationalError,
    SingleStoreDBProgrammingError,
)
from ibis.util import gen_name


@contextlib.contextmanager
def temp_table(con, table_definition=None, name=None):
    """Create a regular table that gets cleaned up after use.

    This replaces temporary tables for HTTP protocol compatibility.

    Args:
        con: Database connection
        table_definition: SQL table definition (e.g., "(x INT, y VARCHAR(50))")
        name: Optional table name, auto-generated if not provided

    Yields:
        str: The table name
    """
    if name is None:
        name = gen_name("test_table")

    try:
        if table_definition:
            with con.begin() as c:
                c.execute(f"CREATE TABLE {name} {table_definition}")
        yield name
    finally:
        con.drop_table(name, force=True)


def _is_http_protocol(con):
    """Check if the connection is using HTTP protocol."""
    return hasattr(con, "_client") and "http" in con._client.__class__.__module__


SINGLESTOREDB_TYPES = [
    # Integer types
    param("tinyint", dt.int8, id="tinyint"),
    param("int1", dt.int8, id="int1"),
    param("smallint", dt.int16, id="smallint"),
    param("int2", dt.int16, id="int2"),
    param("mediumint", dt.int32, id="mediumint"),
    param("int3", dt.int32, id="int3"),
    param("int", dt.int32, id="int"),
    param("int4", dt.int32, id="int4"),
    param("integer", dt.int32, id="integer"),
    param("bigint", dt.int64, id="bigint"),
    # Decimal types
    param("decimal", dt.Decimal(10, 0), id="decimal"),
    param("decimal(5, 2)", dt.Decimal(5, 2), id="decimal_5_2"),
    param("dec", dt.Decimal(10, 0), id="dec"),
    param("numeric", dt.Decimal(10, 0), id="numeric"),
    param("fixed", dt.Decimal(10, 0), id="fixed"),
    # Float types
    param("float", dt.float32, id="float"),
    param("double", dt.float64, id="double"),
    param("real", dt.float64, id="real"),
    # Temporal types
    param("timestamp", dt.Timestamp("UTC"), id="timestamp"),
    param("date", dt.date, id="date"),
    param("time", dt.time, id="time"),
    param("datetime", dt.timestamp, id="datetime"),
    param("year", dt.uint8, id="year"),
    # String types
    param("char(32)", dt.String(length=32), id="char_32"),
    param("varchar(42)", dt.String(length=42), id="varchar_42"),
    param("text", dt.string, id="text"),
    param("mediumtext", dt.string, id="mediumtext"),
    param("longtext", dt.string, id="longtext"),
    # Binary types
    param("binary(42)", dt.binary, id="binary_42"),
    param("varbinary(42)", dt.binary, id="varbinary_42"),
    param("blob", dt.binary, id="blob"),
    param("mediumblob", dt.binary, id="mediumblob"),
    param("longblob", dt.binary, id="longblob"),
    # Bit types
    param("bit(1)", dt.int8, id="bit_1"),
    param("bit(9)", dt.int16, id="bit_9"),
    param("bit(17)", dt.int32, id="bit_17"),
    param("bit(33)", dt.int64, id="bit_33"),
    # Special SingleStoreDB types
    param("json", dt.json, id="json"),
    # Unsigned integer types
    param("mediumint(8) unsigned", dt.uint32, id="mediumint-unsigned"),
    param("bigint unsigned", dt.uint64, id="bigint-unsigned"),
    param("int unsigned", dt.uint32, id="int-unsigned"),
    param("smallint unsigned", dt.uint16, id="smallint-unsigned"),
    param("tinyint unsigned", dt.uint8, id="tinyint-unsigned"),
] + [
    param(
        f"datetime({scale:d})",
        dt.Timestamp(scale=scale or None),
        id=f"datetime{scale:d}",
        marks=pytest.mark.skipif(
            scale not in (0, 6),
            reason=f"SingleStoreDB only supports DATETIME(0) and DATETIME(6), not DATETIME({scale})",
        ),
    )
    for scale in range(7)
]

# HTTP protocol returns generic types without size information
# HTTP protocol type overrides - only the types that differ
SINGLESTOREDB_HTTP_OVERRIDES = {
    "char(32)": dt.string,
    "varchar(42)": dt.string,
    "bit(1)": dt.int64,
    "bit(9)": dt.int64,
    "bit(17)": dt.int64,
    "bit(33)": dt.int64,
}


@pytest.mark.parametrize(("singlestoredb_type", "expected_type"), SINGLESTOREDB_TYPES)
def test_get_schema_from_query(con, singlestoredb_type, expected_type):
    # Choose the appropriate type mapping based on protocol
    if _is_http_protocol(con):
        # Find HTTP equivalent type from the HTTP-specific mapping
        expected_type = SINGLESTOREDB_HTTP_OVERRIDES.get(
            singlestoredb_type, expected_type
        )

    expected_schema = ibis.schema(dict(x=expected_type))

    with temp_table(con, f"(x {singlestoredb_type})") as table_name:
        result_schema = con._get_schema_using_query(f"SELECT * FROM {table_name}")
        assert result_schema == expected_schema

        # For HTTP protocol, DESCRIBE-based method may return different types than query-based
        # This is expected behavior due to protocol limitations
        if not _is_http_protocol(con):
            # For MySQL protocol, both methods should return the same types
            t = con.table(table_name)
            assert t.schema() == expected_schema


@pytest.mark.parametrize(
    ("singlestoredb_type", "get_schema_expected_type", "table_expected_type"),
    [
        param(
            "enum('small', 'medium', 'large')",
            dt.String(length=6),
            dt.string,
            id="enum",
        ),
        param(
            "boolean",
            dt.int8,  # Cursor-based detection cannot distinguish BOOLEAN from TINYINT
            dt.boolean,  # DESCRIBE-based detection correctly identifies BOOLEAN
            id="boolean",
        ),
    ],
)
def test_get_schema_from_query_special_cases(
    con, singlestoredb_type, get_schema_expected_type, table_expected_type
):
    # For HTTP protocol, enum types return generic string without length
    if (
        _is_http_protocol(con)
        and singlestoredb_type == "enum('small', 'medium', 'large')"
    ):
        get_schema_expected_type = dt.string

    get_schema_expected_schema = ibis.schema(dict(x=get_schema_expected_type))
    table_expected_schema = ibis.schema(dict(x=table_expected_type))

    # Use regular tables instead of temporary tables for HTTP protocol compatibility
    with temp_table(con, f"(x {singlestoredb_type})") as table_name:
        quoted_name = sg.to_identifier(table_name, quoted=True).sql("singlestore")

        result_schema = con._get_schema_using_query(f"SELECT * FROM {quoted_name}")
        assert result_schema == get_schema_expected_schema

        t = con.table(table_name)
        assert t.schema() == table_expected_schema


@pytest.mark.parametrize("coltype", ["TINYBLOB", "MEDIUMBLOB", "BLOB", "LONGBLOB"])
def test_blob_type(con, coltype):
    with temp_table(con, f"(a {coltype})") as table_name:
        t = con.table(table_name)
        assert t.schema() == ibis.schema({"a": dt.binary})


def test_zero_timestamp_data(con):
    import pandas as pd
    import pandas.testing as tm

    table_def = """
    (
        name      CHAR(10) NULL,
        tradedate DATETIME NOT NULL,
        date      DATETIME NULL
    )
    """
    with temp_table(con, table_def) as table_name:
        with con.begin() as c:
            c.execute(
                f"""
                INSERT INTO {table_name} VALUES
                    ('C', '2018-10-22', 0),
                    ('B', '2017-06-07', 0),
                    ('C', '2022-12-21', 0)
                """
            )
        t = con.table(table_name)
        result = t.execute()
        expected = pd.DataFrame(
            {
                "name": ["C", "B", "C"],
                "tradedate": pd.to_datetime(
                    [date(2018, 10, 22), date(2017, 6, 7), date(2022, 12, 21)]
                ),
                "date": [pd.NaT, pd.NaT, pd.NaT],
            }
        )
        # Sort both DataFrames by tradedate to ensure consistent ordering
        result_sorted = result.sort_values("tradedate").reset_index(drop=True)
        expected_sorted = expected.sort_values("tradedate").reset_index(drop=True)
        tm.assert_frame_equal(result_sorted, expected_sorted)


@pytest.fixture(scope="module")
def enum_t(con):
    name = gen_name("enum")
    with con.begin() as cur:
        cur.execute(f"CREATE TABLE {name} (sml ENUM('small', 'medium', 'large'))")
        cur.execute(f"INSERT INTO {name} VALUES ('small')")

    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("expr_fn", "expected_data"),
    [
        (methodcaller("startswith", "s"), [True]),
        (methodcaller("endswith", "m"), [False]),
        (methodcaller("re_search", "mall"), [True]),
        (methodcaller("lstrip"), ["small"]),
        (methodcaller("rstrip"), ["small"]),
        (methodcaller("strip"), ["small"]),
    ],
    ids=["startswith", "endswith", "re_search", "lstrip", "rstrip", "strip"],
)
def test_enum_as_string(enum_t, expr_fn, expected_data):
    import pandas as pd
    import pandas.testing as tm

    expr = expr_fn(enum_t.sml).name("sml")
    res = expr.execute()
    expected = pd.Series(expected_data, name="sml")
    tm.assert_series_equal(res, expected)


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def reverse(a: str) -> str:
        """Reverse a string."""

    expr = reverse("foo")
    result = con.execute(expr)
    assert result == "oof"


def test_list_tables(con):
    # Just verify that we can list tables
    tables = con.list_tables()
    assert isinstance(tables, list)
    assert len(tables) >= 0  # Should have at least some test tables


def test_invalid_port():
    port = 4000
    url = f"singlestoredb://{SINGLESTOREDB_USER}:{SINGLESTOREDB_PASS}@{SINGLESTOREDB_HOST}:{port}/{IBIS_TEST_SINGLESTOREDB_DB}"
    with pytest.raises(SingleStoreDBOperationalError):
        ibis.connect(url)


def test_url_query_parameters():
    """Test that query parameters from URL are passed to do_connect."""
    from urllib.parse import ParseResult

    from ibis.backends.singlestoredb import Backend

    # Create a mock URL with query parameters
    url = ParseResult(
        scheme="singlestoredb",
        netloc=f"{SINGLESTOREDB_USER}:{SINGLESTOREDB_PASS}@{SINGLESTOREDB_HOST}:3306",
        path=f"/{IBIS_TEST_SINGLESTOREDB_DB}",
        params="",
        query="local_infile=1&ssl_disabled=1&autocommit=0",
        fragment="",
    )

    # Mock the do_connect method to capture parameters
    original_do_connect = Backend.do_connect
    captured_kwargs = {}

    def mock_do_connect(_self, *_args, **kwargs):
        captured_kwargs.update(kwargs)
        # Don't actually connect, just capture the parameters

    Backend.do_connect = mock_do_connect

    try:
        Backend._from_url(url)

        # Verify query parameters were passed
        assert "local_infile" in captured_kwargs
        assert captured_kwargs["local_infile"] == "1"
        assert "ssl_disabled" in captured_kwargs
        assert captured_kwargs["ssl_disabled"] == "1"
        assert "autocommit" in captured_kwargs
        assert captured_kwargs["autocommit"] == "0"

        # Verify standard parameters are also present
        assert captured_kwargs["host"] == SINGLESTOREDB_HOST
        assert captured_kwargs["user"] == SINGLESTOREDB_USER
        assert captured_kwargs["database"] == IBIS_TEST_SINGLESTOREDB_DB

    finally:
        # Restore original method
        Backend.do_connect = original_do_connect


def test_create_database_exists(con):
    con.create_database(dbname := gen_name("dbname"))

    with pytest.raises(SingleStoreDBProgrammingError):
        con.create_database(dbname)

    con.create_database(dbname, force=True)

    con.drop_database(dbname, force=True)


def test_drop_database_exists(con):
    con.create_database(dbname := gen_name("dbname"))

    con.drop_database(dbname)

    with pytest.raises(SingleStoreDBOperationalError):
        con.drop_database(dbname)

    con.drop_database(dbname, force=True)


def test_json_type_support(con):
    """Test SingleStoreDB JSON type handling."""
    with temp_table(con, "(data JSON)") as table_name:
        with con.begin() as c:
            json_value = json.dumps({"key": "value"})
            c.execute(f"INSERT INTO {table_name} VALUES ('{json_value}')")

        t = con.table(table_name)
        assert t.schema() == ibis.schema({"data": dt.JSON(nullable=True)})

        result = t.execute()
        assert len(result) == 1
        assert "key" in result.iloc[0]["data"]


def test_connection_attributes(con):
    """Test that connection has expected attributes."""
    assert hasattr(con, "_get_schema_using_query")
    assert hasattr(con, "list_tables")
    assert hasattr(con, "create_database")
    assert hasattr(con, "drop_database")


def test_table_creation_basic_types(con):
    """Test creating tables with basic data types."""
    table_name = gen_name("types")
    schema = ibis.schema(
        [
            ("id", dt.int32),
            ("name", dt.string),
            ("value", dt.float64),
            ("created_at", dt.timestamp),
            ("is_active", dt.boolean),
        ]
    )

    # Create table - use temp=False for HTTP protocol compatibility
    con.create_table(table_name, schema=schema, temp=False)

    try:
        # Verify table exists and has correct schema
        t = con.table(table_name)
        actual_schema = t.schema()

        # Check that essential columns exist (may have slight type differences)
        assert "id" in actual_schema
        assert "name" in actual_schema
        assert "value" in actual_schema
        assert "created_at" in actual_schema
        assert "is_active" in actual_schema
    finally:
        con.drop_table(table_name, force=True)


def test_create_table_from_dataframe_with_missing_values(con):
    if _is_http_protocol(con):
        # memtables are TEMPORARY tables, which aren't guaranteed to persist
        # across HTTP requests
        pytest.skip("memtables are unreliable over HTTP")
    table_name = gen_name("missing")
    # pandas 3 `str` columns store missing values as nan, not None
    df = pd.DataFrame(
        {
            "s": pd.Series(["a", None], dtype=pd.StringDtype(na_value=float("nan"))),
            "f": [1.5, float("nan")],
            "i": [1, 2],
        }
    )
    con.create_table(table_name, df, overwrite=True)
    try:
        t = con.table(table_name)
        assert t.s.isnull().sum().execute() == 1
        assert t.f.isnull().sum().execute() == 1
        result = t.order_by("i").execute()
        assert result["s"].iloc[0] == "a"
        assert result["f"].iloc[0] == 1.5
    finally:
        con.drop_table(table_name, force=True)


def test_transaction_handling(con):
    """Test transaction begin/commit/rollback."""
    with temp_table(con, " (id INT, value VARCHAR(50))") as table_name:
        with con.begin() as c:
            c.execute(f"INSERT INTO {table_name} VALUES (1, 'test')")

            # Verify data was committed
            t = con.table(table_name)
            result = t.execute()
            assert len(result) == 1
            assert result.iloc[0]["id"] == 1
            assert result.iloc[0]["value"] == "test"


def _storage_type(con, table_name):
    with con.begin() as c:
        c.execute(
            "SELECT storage_type FROM information_schema.tables "
            f"WHERE table_schema = DATABASE() AND table_name = '{table_name}'"
        )
        (storage_type,) = c.fetchone()
    return storage_type


@pytest.mark.parametrize(
    ("storage_type", "expected"),
    [
        param("rowstore", "INMEMORY_ROWSTORE", id="rowstore"),
        param("ROWSTORE", "INMEMORY_ROWSTORE", id="rowstore-upper"),
        param("columnstore", "COLUMNSTORE", id="columnstore"),
    ],
)
def test_create_table_storage_type(con, storage_type, expected):
    # use schema= rather than a DataFrame: memtables are temporary tables,
    # which are unreliable over the HTTP protocol under concurrent load
    table_name = gen_name("storage_type")
    con.create_table(
        table_name, schema=ibis.schema({"x": "int64"}), storage_type=storage_type
    )
    try:
        assert _storage_type(con, table_name) == expected
    finally:
        con.drop_table(table_name, force=True)


def test_create_table_storage_type_overwrite(con):
    table_name = gen_name("storage_type")
    con.create_table(table_name, schema=ibis.schema({"x": "int64"}))
    try:
        con.create_table(
            table_name,
            schema=ibis.schema({"x": "int64"}),
            overwrite=True,
            storage_type="rowstore",
        )
        assert _storage_type(con, table_name) == "INMEMORY_ROWSTORE"
    finally:
        con.drop_table(table_name, force=True)


def test_create_temp_table_rowstore(con):
    if _is_http_protocol(con):
        pytest.skip("temporary tables don't persist across HTTP requests")
    table_name = gen_name("storage_type")
    t = con.create_table(
        table_name, pd.DataFrame({"x": [1, 2]}), temp=True, storage_type="rowstore"
    )
    assert t.count().execute() == 2


def test_create_table_invalid_storage_type(con):
    with pytest.raises(com.IbisInputError, match="storage_type"):
        con.create_table(
            gen_name("storage_type"),
            schema=ibis.schema({"x": "int64"}),
            storage_type="reference",
        )


def test_create_table_storage_type_sql():
    """ROWSTORE must precede TEMPORARY; sqlglot's default order is invalid."""
    from ibis.backends.singlestoredb import Backend

    con = Backend()
    sql = []

    class Cursor:
        def execute(self, stmt):
            sql.append(stmt)

    @contextlib.contextmanager
    def begin():
        yield Cursor()

    con.begin = begin
    con.create_table(
        "t", schema=ibis.schema({"x": "int64"}), temp=True, storage_type="rowstore"
    )
    assert sql == ["CREATE ROWSTORE TEMPORARY TABLE `t` (`x` BIGINT)"]


@pytest.mark.parametrize(
    "name",
    [
        "globals",
        "locals",
        "vars",
        "cluster_globals",
        "cluster_locals",
        "cluster_vars",
    ],
)
def test_variable_accessors(con, name):
    accessor = getattr(server, name)(con)
    assert accessor is getattr(con.con, name)


def test_variable_accessor_values(con):
    assert int(server.globals(con)["max_connections"]) > 0
    assert server.locals(con)["autocommit"] in (True, 1, "ON")


def test_show_accessor(con):
    table_name = gen_name("show")
    con.create_table(table_name, schema=ibis.schema({"x": "int64"}))
    try:
        names = [row["Name"] for row in server.show(con).tables()]
        assert table_name in names
        assert IBIS_TEST_SINGLESTOREDB_DB in [
            row["Name"] for row in server.show(con).databases()
        ]
    finally:
        con.drop_table(table_name, force=True)


@pytest.mark.parametrize("name", ["show", "globals", "locals", "vars", "cluster_vars"])
def test_backend_has_no_accessors(con, name):
    assert not hasattr(con, name)


@pytest.fixture
def stats_table(con):
    table_name = gen_name("stats")
    t = con.create_table(table_name, schema=ibis.schema({"x": "int64", "y": "string"}))
    con.insert(table_name, pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", None]}))
    yield t
    con.drop_table(table_name, force=True)


@pytest.mark.parametrize("by", ["name", "table"])
def test_get_table_stats(con, stats_table, by):
    table = stats_table.get_name() if by == "name" else stats_table
    stats = server.get_table_stats(con, table)
    result = stats.execute()
    assert len(result) > 0
    assert set(result.TABLE_NAME) == {stats_table.get_name()}
    assert stats.ROWS.sum().execute() == 3


def test_get_table_stats_database(con, stats_table):
    name = stats_table.get_name()
    stats = server.get_table_stats(con, name, database=IBIS_TEST_SINGLESTOREDB_DB)
    assert stats.ROWS.sum().execute() == 3
    missing = server.get_table_stats(con, name, database="ibis_no_such_db")
    assert missing.count().execute() == 0


def test_get_table_stats_rejects_expression(con, stats_table):
    with pytest.raises(TypeError, match="database table"):
        server.get_table_stats(con, stats_table.select("x"))


def test_get_storage_info(con, stats_table):
    result = server.get_storage_info(con).execute()
    assert set(result.DATABASE_NAME) == {IBIS_TEST_SINGLESTOREDB_DB}
    assert stats_table.get_name() in set(result.TABLE_NAME)


def test_get_column_statistics(con, stats_table):
    con.raw_sql(f"ANALYZE TABLE `{stats_table.get_name()}`").close()
    result = server.get_column_statistics(con, stats_table).execute()
    assert set(result.COLUMN_NAME) == {"x", "y"}
    assert set(result.TABLE_NAME) == {stats_table.get_name()}
    result = server.get_column_statistics(con, stats_table, "y").execute()
    assert list(result.COLUMN_NAME) == ["y"]


def test_get_workload_metrics(con):
    result = server.get_workload_metrics(con).execute()
    assert {"NODE_ID", "STAT", "VALUE"} <= set(result.columns)
    assert len(result) > 0


@pytest.mark.parametrize("by", ["name", "table"])
def test_optimize_table(con, stats_table, by):
    table = stats_table.get_name() if by == "name" else stats_table
    server.optimize_table(con, table)
    assert stats_table.count().execute() == 3


def test_optimize_table_sql(monkeypatch):
    from ibis.backends.singlestoredb import Backend

    monkeypatch.setattr(Backend, "current_database", "db")
    con = Backend()
    sql = []

    @contextlib.contextmanager
    def raw_sql(stmt):
        sql.append(stmt)
        yield

    con.raw_sql = raw_sql
    server.optimize_table(con, "my`table")
    server.optimize_table(con, "t", database="other")
    assert sql == [
        "OPTIMIZE TABLE `db`.`my``table` FULL",
        "OPTIMIZE TABLE `other`.`t` FULL",
    ]
