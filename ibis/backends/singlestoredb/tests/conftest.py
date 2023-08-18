from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from packaging.version import parse as parse_version

import ibis
from ibis.backends.conftest import TEST_TABLES, init_database
from ibis.backends.tests.base import RoundHalfToEven, ServiceBackendTest, UnorderedComparator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

SINGLESTOREDB_USER = os.environ.get("IBIS_TEST_SINGLESTOREDB_USER", "ibis")
SINGLESTOREDB_PASSWORD = os.environ.get("IBIS_TEST_SINGLESTOREDB_PASSWORD", "ibis")
SINGLESTOREDB_HOST = os.environ.get("IBIS_TEST_SINGLESTOREDB_HOST", "localhost")
SINGLESTOREDB_PORT = int(os.environ.get("IBIS_TEST_SINGLESTOREDB_PORT", 3306))
IBIS_TEST_SINGLESTOREDB_DB = os.environ.get("IBIS_TEST_SINGLESTOREDB_DATABASE", "ibis_testing")


class TestConf(UnorderedComparator, ServiceBackendTest, RoundHalfToEven):
    # singlestoredb has the same rounding behavior as postgres
    check_dtype = False
    returned_timestamp_unit = "s"
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    native_bool = False
    supports_structs = False
    supports_window_operations = False
    service_name = "singlestoredb"
    deps = "singlestoredb", "sqlalchemy", "sqlalchemy_singlestoredb"

    @property
    def test_files(self) -> Iterable[Path]:
        return self.data_dir.joinpath("csv").glob("*.csv")

    def _load_data(
        self,
        *,
        user: str = SINGLESTOREDB_USER,
        password: str = SINGLESTOREDB_PASSWORD,
        host: str = SINGLESTOREDB_HOST,
        port: int = SINGLESTOREDB_PORT,
        database: str = IBIS_TEST_SINGLESTOREDB_DB,
        **_: Any,
    ) -> None:
        """Load test data into a SingleStoreDB backend instance.

        Parameters
        ----------
        data_dir
            Location of testdata
        script_dir
            Location of scripts defining schemas
        """
        engine = init_database(
            url=sa.engine.make_url(
                f"singlestoredb://{user}:{password}@{host}:{port:d}?local_infile=1",
            ),
            database=database,
            schema=self.ddl_script,
            isolation_level="READ COMMITTED",
            recreate=False,
        )
        with engine.begin() as con:
            for table in TEST_TABLES:
                csv_path = self.data_dir / "csv" / f"{table}.csv"
                lines = [
                    f"LOAD DATA LOCAL INFILE {str(csv_path)!r}",
                    f"INTO TABLE {table}",
                    "COLUMNS TERMINATED BY ','",
                    """OPTIONALLY ENCLOSED BY '"'""",
                    "LINES TERMINATED BY '\\n'",
                    "NULL DEFINED BY ''",
                    "IGNORE 1 LINES",
                ]
                con.exec_driver_sql("\n".join(lines))

    @staticmethod
    def connect(*, tmpdir, worker_id, **kw):
        return ibis.singlestoredb.connect(
            host=SINGLESTOREDB_HOST,
            user=SINGLESTOREDB_USER,
            password=SINGLESTOREDB_PASSWORD,
            database=IBIS_TEST_SINGLESTOREDB_DB,
            port=SINGLESTOREDB_PORT,
            **kw,
        )


@pytest.fixture(scope="session")
def setup_privs():
    engine = sa.create_engine(f"singlestoredb://root:@{SINGLESTOREDB_HOST}:{SINGLESTOREDB_PORT:d}")
    with engine.begin() as con:
        # allow the ibis user to use any database
        con.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS `{IBIS_TEST_SINGLESTOREDB_DB}`")
        con.exec_driver_sql(
            f"GRANT CREATE,SELECT,DROP ON `{IBIS_TEST_SINGLESTOREDB_DB}`.* TO `{SINGLESTOREDB_USER}`@`%`"
        )
    yield
    with engine.begin() as con:
        con.exec_driver_sql(f"DROP DATABASE IF EXISTS `{IBIS_TEST_SINGLESTOREDB_DB}`")


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_dir, worker_id):
    return TestConf.load_data(data_dir, tmp_path_factory, worker_id).connection


@pytest.fixture(scope="session")
def con_nodb():
    return ibis.singlestoredb.connect(
        host=SINGLESTOREDB_HOST, user=SINGLESTOREDB_USER, password=SINGLESTOREDB_PASSWORD, port=SINGLESTOREDB_PORT
    )
