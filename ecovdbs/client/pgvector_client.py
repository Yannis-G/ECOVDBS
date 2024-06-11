import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, sql, Cursor

from .base_client import BaseClient, BaseIndexConfig
from .pgvector_config import PgvectorConfig


class PgvectorClient(BaseClient):

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: dict | None = None) -> None:
        if db_config is None:
            db_config = PgvectorConfig().to_dict()

        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config
        self.__table_name = "ecovdbs"
        self.__index_name = "idx:ecovdbs"
        self.__id_name = "id"
        self.__vector_name = "vector"

        self.__conn: Connection = psycopg.connect(
            f"host={self.__db_config['host']} port={self.__db_config['port']} dbname={self.__db_config['dbname']} user={self.__db_config['user']} password={self.__db_config['password']}")
        self.__conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.__conn.commit()

        register_vector(self.__conn)

        drop_table = sql.SQL("DROP TABLE IF EXISTS {table_name};").format(table_name=sql.Identifier(self.__table_name))
        self.__conn.execute(drop_table)
        self.__conn.commit()

        drop_index = sql.SQL("DROP INDEX IF EXISTS {index_name};").format(index_name=sql.Identifier(self.__index_name))
        self.__conn.execute(drop_index)
        self.__conn.commit()

        create_table = (sql.SQL(
            "CREATE TABLE IF NOT EXISTS {table_name} ({id_name} bigserial PRIMARY KEY, {vector_name} vector({dimension}));").format(
            table_name=sql.Identifier(self.__table_name), id_name=sql.Identifier(self.__id_name),
            vector_name=sql.Identifier(self.__vector_name), dimension=dimension))
        self.__conn.execute(create_table)
        self.__conn.commit()

    def insert(self, embeddings: list[list[float]]) -> None:
        cur: Cursor = self.__conn.cursor()
        with cur.copy(sql.SQL("COPY {table_name} ({id_name}, {vector_name}) FROM STDIN (FORMAT BINARY)").format(
                table_name=sql.Identifier(self.__table_name), id_name=sql.Identifier(self.__id_name),
                vector_name=sql.Identifier(self.__vector_name))) as copy:
            copy.set_types(["bigint", "vector"])
            for i, embedding in enumerate(embeddings):
                copy.write_row((i, embedding))
        self.__conn.commit()

    def batch_insert(self, embeddings: list[list[float]]) -> None:
        pass

    def create_index(self) -> None:
        index_param = self.__index_config.index_param()
        self.__set_param(index_param["set"])
        opt = []
        if index_param["with"]:
            for k, v in index_param["with"].items():
                opt.append(sql.SQL("{key} = {val}").format(key=sql.Identifier(k), val=sql.Literal(v)))
            with_clause = sql.SQL(" WITH ({});").format(sql.SQL(", ").join(opt))
        else:
            with_clause = sql.Composed(())
        index_clause = sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING {index_type} ({vector_name} {metric_type})").format(
            index_name=sql.Identifier(self.__index_name), table_name=sql.Identifier(self.__table_name),
            index_type=sql.Identifier(index_param["index_type"]), vector_name=sql.Identifier(self.__vector_name),
            metric_type=sql.Identifier(index_param["metric_type"]))
        index_with_with = (index_clause + with_clause)
        self.__conn.execute(index_with_with)
        self.__conn.commit()

    def __set_param(self, param):
        if param:
            for k, v in param.items():
                command = sql.SQL("SET {key} = {val}").format(key=sql.Identifier(k), val=sql.Literal(v))
                self.__conn.execute(command)
            self.__conn.commit()

    def disk_storage(self):
        pass

    def index_storage(self):
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        search_param = self.__index_config.search_param()
        self.__set_param(search_param["set"])
        select = sql.Composed([
            sql.SQL(
                "SELECT {id_name} FROM {table_name} ORDER BY {vector_name} ").format(
                id_name=sql.Identifier(self.__id_name), table_name=sql.Identifier(self.__table_name),
                vector_name=sql.Identifier(self.__vector_name)),
            sql.SQL(search_param["metric_operator"]),
            sql.SQL(" %s::vector LIMIT %s::int")
        ])
        res = self.__conn.execute(select, (query, k))
        return [int(r[0]) for r in res.fetchall()]
