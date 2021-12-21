import contextlib
import re
import warnings

import sqlalchemy as sa
import sqlalchemy.dialects.mysql as singlestore

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.base.sql.registry.helpers import quote_identifier

from ibis.expr.types import AnyColumn

from .compiler import SingleStoreCompiler

from . import ddl, udf
from .udf import (  # noqa F408
    aggregate_function,
    scalar_function,
    wrap_uda,
    wrap_udf,
    SingleStoreUDF,
)


# TODO: Patch in an `apply` method for demo

def apply(self, func, axis=0, raw=False, result_type=None, args=None, **kwargs):
    args = args or []
    name = func.__name__
    func = self.op().table.op().source.create_function(func)
    out = func(self, *args)
    #self.op().table.op().source.raw_sql(f'drop function {name}')
    return out

AnyColumn.apply = apply


class FuncDict(dict):

    def __init__(self, con):
        super(dict, self).__init__()
        self.__dict__['_con'] = con
        self.__dict__['_database_name'] = con.database_name
        self._refresh()

    def __call__(self, refresh=False):
        if refresh:
            self._refresh()
        return self

    def _refresh(self):
        self.clear()
        db = quote_identifier(self._database_name)
        for item in self._con.raw_sql(f'show functions in {db}').fetchall():
            self[item[0]] = self._make_func(item[0])

    def _has_function(self, name):
        db = quote_identifier(self._database_name)
        qname = quote_identifier(item[0])
        funcs = self._con.raw_sql(f'show functions in {db} like {name}').fetchall()
        if len(funcs) == 1:
            return True
        if len(funcs) == 0:
            return False
        raise ValueError('More than one function matches name: {}'.format(', '.join(funcs)))

    def _make_func(self, name):
        db = quote_identifier(self._database_name)
        qname = quote_identifier(name)
        proto = self._con.raw_sql(f'show create function {db}.{qname}').fetchall()[0][2]
        proto = re.split(r'\bfunction\s+', proto, flags=re.I)[-1]
        name, proto = proto.split('(', 1)
        if re.search(r'\)\s+returns\s+', proto, flags=re.I):
            sig, ret = re.split(r'\)\s+returns\s+', proto, flags=re.I)
            ret, ftype = re.split(r'\s+as\s+', ret, flags=re.I)
        else:
            ret = None
            sig, ftype = re.split(r'\s+as\s+', proto, flags=re.I)
        ftype, info = ftype.split("'", 1)
        ftype = ftype.strip()
        m = re.search(r"^(.*)'\s+format\s+(\w+)\s*;\s*$", info, flags=re.I)
        code = m.group(1)
        format = m.group(2)
        if name.startswith('`'):
            name = name[1:-1]
        input_names = [re.match(r'^\s*(\w+)\s+\w+', x).group(1) for x in sig.split(',')]
        inputs = [re.match(r'^\s*\w+\s+(\w+)', x).group(1) for x in sig.split(',')]
        nullable = [not re.search(r'\bnot\s+null\b', x, flags=re.I) for x in sig.split(',')]
        inputs = [dict(bigint='int64', text='string')[x] for x in inputs]
        out_nullable = False
        output = ret
        if output:
            out_nullable = not re.search(r'\bnot\s+null\b', output, flags=re.I)
            output = re.match(r'^\s*(\w+)', output).group(1)
            output = dict(bigint='int64', text='string')[output]
        func = SingleStoreUDF(inputs, output, name)
        func.__doc__ = self._make_func_doc(name, ftype.lower(),
                                           list(zip(input_names, inputs, nullable)),
                                           (output, out_nullable), code, format)
        func.register(name, self._database_name)
        return func

    def _make_func_doc(self, name, ftype, inputs, output, code, format):
        doc = [f'Call `{name}` {ftype} function', '']
        if ftype == 'remote service':
            doc.append(f'Accesses remote service at {code} using {format} format.')
            doc.append('')
        doc.extend(['Parameters', '----------'])
        for name, dtype, nullable in inputs:
            arg = f'{name} : {dtype}'
            if nullable:
                arg += ' or None'
            doc.append(arg)
        if output and output[0]:
            doc.append('')
            doc.extend(['Returns', '-------'])
            ret = f'{output[0]}'
            if output[1]:
                ret += ' or None'
            doc.append(ret)
        doc.append('')
        return '\n'.join(doc)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            if self._has_function(name):
                self[name] = self._make_func(name)
                return
            raise AttributeError(f"'dict' object has no attribute '{name}'")

    def __getitem__(self, name):
        try:
            return dict.__getitem__(self, name)
        except:
            if self._has_function(name):
                func = self._make_func(name)
                self[name] = func
                return func
            raise

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)



class TableAccessor(object):

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return self._backend._table(name)

    def __call__(self, name, database=None, schema=None):
        return self._backend._table(name, database=database, schema=schema)


class Backend(BaseAlchemyBackend):
    name = 'singlestore'
    compiler = SingleStoreCompiler

    def connect(
        self,
        host='localhost',
        user=None,
        password=None,
        port=3306,
        database=None,
        url=None,
        driver='pymysql'
    ):

        """Create an Ibis client located at `user`:`password`@`host`:`port`
        connected to a SingleStore database named `database`.

        Parameters
        ----------
        host : string, default 'localhost'
        user : string, default None
        password : string, default None
        port : string or integer, default 3306
        database : string, default None
        url : string, default None
            Complete SQLAlchemy connection string. If passed, the other
            connection arguments are ignored.
        driver : string, default 'pymysql'

        Returns
        -------
        Backend

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> host = os.environ.get('IBIS_TEST_SINGLESTORE_HOST', 'localhost')
        >>> user = os.environ.get('IBIS_TEST_SINGLESTOER_USER', getpass.getuser())
        >>> password = os.environ.get('IBIS_TEST_SINGLESTORE_PASSWORD')
        >>> database = os.environ.get('IBIS_TEST_SINGLESTORE_DATABASE',
        ...                           'ibis_testing')
        >>> con = connect(
        ...     database=database,
        ...     host=host,
        ...     user=user,
        ...     password=password
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table('functional_alltypes')
        >>> t
        SingleStoreTable[table]
          name: functional_alltypes
          schema:
            index : int64
            Unnamed: 0 : int64
            id : int32
            bool_col : int8
            tinyint_col : int8
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32
        """
        if driver != 'pymysql':
            raise NotImplementedError(
                'pymysql is currently the only supported driver'
            )
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f'mysql+{driver}',
        )

        new_backend = super().connect(sa.create_engine(alchemy_url))
        new_backend.database_name = alchemy_url.database

        new_backend.table = TableAccessor(new_backend)

        new_backend._funcs = {}

        return new_backend

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_timezone = bind.execute(
                'SELECT @@session.time_zone'
            ).scalar()
            try:
                bind.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Couldn't set singlestore timezone: {str(e)}")

            try:
                yield bind
            finally:
                query = "SET @@session.time_zone = '{}'"
                bind.execute(query.format(previous_timezone))

    def _table(self, name, database=None, schema=None):
        """Create a table expression that references a particular a table
        called `name` in a SingleStore database called `database`.

        Parameters
        ----------
        name : str
            The name of the table to retrieve.
        database : str, optional
            The database in which the table referred to by `name` resides. If
            ``None`` then the ``current_database`` is used.
        schema : str, optional
            The schema in which the table resides.  If ``None`` then the
            `public` schema is assumed.

        Returns
        -------
        table : TableExpr
            A table expression.
        """
        if database is not None and database != self.current_database:
            return self.database(name=database).table(name=name, schema=schema)
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = self.table_class(alch_table, self, self._schemas.get(name))
            return self.table_expr_class(node)

    @property
    def funcs(self):
        if self.database_name not in self._funcs:
            self._funcs[self.database_name] = FuncDict(self)
        return self._funcs[self.database_name]

#   def create_function(self, func, name=None, database=None):
#       """
#       Creates a function within SingleStore
#
#       Parameters
#       ----------
#       func : SingleStoreUDF or SingleStoreUDA
#           Created with wrap_udf or wrap_uda
#       name : string (optional)
#       database : string (optional)
#       """
#       if name is None:
#           name = func.name
#       database = database or self.current_database
#
#       if isinstance(func, udf.SingleStoreUDF):
#           stmt = ddl.CreateUDF(func, name=name, database=database).compile()
#       elif isinstance(func, udf.SingleStoreUDA):
#           stmt = ddl.CreateUDA(func, name=name, database=database).compile()
#       else:
#           raise TypeError(func)
#       self.raw_sql(stmt)

    def create_function(self, func, database=None):
        """
        Create a function within SingleStore from Python source

        Parameters
        ----------
        func : Function
            Python function
        database : string, optional
            Name of the database to upload to. The current database
            is used by default.

        Returns
        -------
        SingleStoreUDF

        """
        import inspect
        import os
        import tempfile

        database = database or self.current_database

        TYPE_MAP = {
            int: 'int64',
            float: 'double',
            str: 'varchar(255)',
            None: None,
        }

        argspec = inspect.getfullargspec(func)
        anno = argspec.annotations
        inputs = []
        output = TYPE_MAP[anno.get('return')]

        for arg in argspec.args:
            inputs.append(TYPE_MAP[anno[arg]])

        with tempfile.TemporaryDirectory() as tmp:
            tmpf = os.path.join(tmp, 'func.py')
            with open(tmpf, 'w') as outfile:
                outfile.write(inspect.getsource(func))

            # Create function object
            out = wrap_udf(tmpf, inputs, output, func.__name__)

            # TODO: Support UDAs too.
            self.raw_sql(ddl.CreateUDF(out, name=out.name, database=database).compile())

            # Register the function with Ibis
            out.register(out.name, database)

            return out


# TODO(kszucs): unsigned integers


@dt.dtype.register((singlestore.DOUBLE, singlestore.REAL))
def singlestore_double(satype, nullable=True):
    return dt.Double(nullable=nullable)


@dt.dtype.register(singlestore.FLOAT)
def singlestore_float(satype, nullable=True):
    return dt.Float(nullable=nullable)


@dt.dtype.register(singlestore.TINYINT)
def singlestore_tinyint(satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(singlestore.BLOB)
def singlestore_blob(satype, nullable=True):
    return dt.Binary(nullable=nullable)
