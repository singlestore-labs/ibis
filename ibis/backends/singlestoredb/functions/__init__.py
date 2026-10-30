"""SingleStoreDB vector and JSON functions that have no Ibis method.

The vector functions are in `functions.vector` and the JSON functions in
`functions.json`. They build Ibis expressions and don't need a connection:

>>> from ibis.backends.singlestoredb import functions as s2  # doctest: +SKIP
>>> t.select(  # doctest: +SKIP
...     score=s2.vector.dot_product(t.embedding, [0.1, 0.2, 0.3]),
...     city=s2.json.extract_string(t.doc, "address", "city"),
... )

Import the package rather than its `json` module, which would hide the
standard library `json` module.

For a user-defined function or aggregate on the server, use
`con.function`. For other builtins, declare a builtin UDF with
`@ibis.udf.scalar.builtin` or `@ibis.udf.agg.builtin`.
"""

from __future__ import annotations

from ibis.backends.singlestoredb.functions import json, vector

__all__ = ["json", "vector"]
