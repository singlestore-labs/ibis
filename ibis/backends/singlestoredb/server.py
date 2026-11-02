"""SingleStoreDB server variables, `SHOW` commands and table maintenance.

Each function takes a SingleStoreDB connection as its first argument:

>>> from ibis.backends.singlestoredb import server  # doctest: +SKIP
>>> server.globals(con)["max_connections"]  # doctest: +SKIP
>>> server.show(con).tables()  # doctest: +SKIP
>>> server.get_table_stats(con, "users").execute()  # doctest: +SKIP

The variable and `SHOW` accessors are those of the `singlestoredb` client.
Variable accessors are mappings that read and set server variables, e.g.
`server.locals(con)["autocommit"] = False`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.expr.operations as ops
import ibis.expr.types as ir

if TYPE_CHECKING:
    from ibis.backends.singlestoredb import Backend

__all__ = [
    "cluster_globals",
    "cluster_locals",
    "cluster_vars",
    "get_column_statistics",
    "get_storage_info",
    "get_table_stats",
    "get_workload_metrics",
    "globals",
    "locals",
    "optimize_table",
    "show",
    "vars",
]


def show(con: Backend, /) -> Any:
    """Return the accessor for `SHOW` commands, e.g. `show(con).tables()`."""
    return con.con.show


def globals(con: Backend, /) -> Any:
    """Return the accessor for global server variables."""
    return con.con.globals


def locals(con: Backend, /) -> Any:
    """Return the accessor for session server variables."""
    return con.con.locals


def cluster_globals(con: Backend, /) -> Any:
    """Return the accessor for cluster-wide global server variables."""
    return con.con.cluster_globals


def cluster_locals(con: Backend, /) -> Any:
    """Return the accessor for cluster-wide session server variables."""
    return con.con.cluster_locals


def vars(con: Backend, /) -> Any:
    """Return the accessor for server variables.

    Session variables are read first, falling back to global variables.
    """
    return con.con.vars


def cluster_vars(con: Backend, /) -> Any:
    """Return the accessor for cluster-wide server variables."""
    return con.con.cluster_vars


def _info(con: Backend, name: str) -> ir.Table:
    return con.table(name, database="information_schema")


def _table_name(table: str | ir.Table, database: str | None) -> tuple[str, str | None]:
    """Return the name and database of `table`, a name or a table expression."""
    if isinstance(table, str):
        return table, database
    op = table.op()
    if not isinstance(op, ops.DatabaseTable):
        raise TypeError(
            f"Expected a table name or a database table, got {type(op).__name__}"
        )
    return op.name, database or op.namespace.database


def get_storage_info(con: Backend, /, database: str | None = None) -> ir.Table:
    """Return storage statistics for the tables in a database.

    The result has a row for each partition of each table, from
    `information_schema.TABLE_STATISTICS`.

    Parameters
    ----------
    con
        A SingleStoreDB connection.
    database
        Database name. Defaults to the current database.
    """
    t = _info(con, "TABLE_STATISTICS")
    return t.filter(t.DATABASE_NAME == (database or con.current_database))


def get_table_stats(
    con: Backend, /, table: str | ir.Table, *, database: str | None = None
) -> ir.Table:
    """Return storage statistics for a table.

    The result has a row for each partition of the table, from
    `information_schema.TABLE_STATISTICS`, e.g. `ROWS` and `MEMORY_USE`.

    Parameters
    ----------
    con
        A SingleStoreDB connection.
    table
        Table name or table expression.
    database
        Database name. Defaults to the table's database, or the current
        database.

    Examples
    --------
    >>> stats = server.get_table_stats(con, "users")  # doctest: +SKIP
    >>> stats.ROWS.sum().execute()  # doctest: +SKIP
    """
    name, database = _table_name(table, database)
    t = get_storage_info(con, database)
    return t.filter(t.TABLE_NAME == name)


def get_column_statistics(
    con: Backend,
    /,
    table: str | ir.Table,
    column: str | None = None,
    *,
    database: str | None = None,
) -> ir.Table:
    """Return optimizer statistics for the columns of a table.

    The result has a row for each column, from
    `information_schema.OPTIMIZER_STATISTICS`, e.g. `ROW_COUNT`, `NULL_COUNT`
    and `CARDINALITY`. Statistics are collected by `ANALYZE TABLE` or
    automatically by the server.

    Parameters
    ----------
    con
        A SingleStoreDB connection.
    table
        Table name or table expression.
    column
        Column name. Defaults to all columns.
    database
        Database name. Defaults to the table's database, or the current
        database.
    """
    name, database = _table_name(table, database)
    t = _info(con, "OPTIMIZER_STATISTICS")
    t = t.filter(
        t.DATABASE_NAME == (database or con.current_database), t.TABLE_NAME == name
    )
    if column is not None:
        t = t.filter(t.COLUMN_NAME == column)
    return t


def get_workload_metrics(con: Backend, /) -> ir.Table:
    """Return workload management statistics for each node.

    The result is `information_schema.MV_WORKLOAD_MANAGEMENT_STATUS`, which
    has a row for each node and statistic.
    """
    return _info(con, "MV_WORKLOAD_MANAGEMENT_STATUS")


def optimize_table(
    con: Backend, /, table: str | ir.Table, *, database: str | None = None
) -> None:
    """Run a full `OPTIMIZE TABLE` on a table.

    Parameters
    ----------
    con
        A SingleStoreDB connection.
    table
        Table name or table expression.
    database
        Database name. Defaults to the table's database, or the current
        database.
    """
    name, database = _table_name(table, database)
    target = sg.table(name, db=database or con.current_database, quoted=True)
    with con.raw_sql(f"OPTIMIZE TABLE {target.sql(con.dialect)} FULL"):
        pass
