"""SingleStoreDB server variables and `SHOW` commands.

Each function takes a SingleStoreDB connection as its first argument:

>>> from ibis.backends.singlestoredb import server  # doctest: +SKIP
>>> server.globals(con)["max_connections"]  # doctest: +SKIP
>>> server.show(con).tables()  # doctest: +SKIP

The variable and `SHOW` accessors are those of the `singlestoredb` client.
Variable accessors are mappings that read and set server variables, e.g.
`server.locals(con)["autocommit"] = False`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ibis.backends.singlestoredb import Backend

__all__ = [
    "cluster_globals",
    "cluster_locals",
    "cluster_vars",
    "globals",
    "locals",
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
