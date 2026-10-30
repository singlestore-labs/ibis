"""SingleStoreDB vector and JSON functions that have no Ibis method.

These functions build Ibis expressions and don't need a connection:

>>> from ibis.backends.singlestoredb import functions as s2  # doctest: +SKIP
>>> t.select(score=s2.dot_product(t.embedding, [0.1, 0.2, 0.3]))  # doctest: +SKIP

The vector functions are in `functions.vector` and the JSON functions in
`functions.json`; both are available from `functions`.

For a user-defined function or aggregate on the server, use
`con.function`. For other builtins, declare a builtin UDF with
`@ibis.udf.scalar.builtin` or `@ibis.udf.agg.builtin`.
"""

from __future__ import annotations

from ibis.backends.singlestoredb.functions import json, vector
from ibis.backends.singlestoredb.functions.json import (
    json_agg,
    json_array_contains_double,
    json_array_contains_json,
    json_array_contains_string,
    json_array_push_double,
    json_array_push_json,
    json_array_push_string,
    json_build_array,
    json_build_object,
    json_delete_key,
    json_exclude_mask,
    json_extract_bigint,
    json_extract_double,
    json_extract_json,
    json_extract_string,
    json_get_type,
    json_include_mask,
    json_keys,
    json_length,
    json_match_any,
    json_match_any_exists,
    json_merge_patch,
    json_pretty,
    json_set_double,
    json_set_json,
    json_set_string,
    json_splice_double,
    json_splice_json,
    json_splice_string,
    reduce,
    to_json,
)
from ibis.backends.singlestoredb.functions.vector import (
    dot_product,
    euclidean_distance,
    json_array_pack,
    json_array_unpack,
    scalar_vector_mul,
    vector_add,
    vector_elements_sum,
    vector_kth_element,
    vector_mul,
    vector_num_elements,
    vector_sort,
    vector_sub,
    vector_subvector,
    vector_sum,
)

__all__ = [
    "dot_product",
    "euclidean_distance",
    "json",
    "json_agg",
    "json_array_contains_double",
    "json_array_contains_json",
    "json_array_contains_string",
    "json_array_pack",
    "json_array_push_double",
    "json_array_push_json",
    "json_array_push_string",
    "json_array_unpack",
    "json_build_array",
    "json_build_object",
    "json_delete_key",
    "json_exclude_mask",
    "json_extract_bigint",
    "json_extract_double",
    "json_extract_json",
    "json_extract_string",
    "json_get_type",
    "json_include_mask",
    "json_keys",
    "json_length",
    "json_match_any",
    "json_match_any_exists",
    "json_merge_patch",
    "json_pretty",
    "json_set_double",
    "json_set_json",
    "json_set_string",
    "json_splice_double",
    "json_splice_json",
    "json_splice_string",
    "reduce",
    "scalar_vector_mul",
    "to_json",
    "vector",
    "vector_add",
    "vector_elements_sum",
    "vector_kth_element",
    "vector_mul",
    "vector_num_elements",
    "vector_sort",
    "vector_sub",
    "vector_subvector",
    "vector_sum",
]
