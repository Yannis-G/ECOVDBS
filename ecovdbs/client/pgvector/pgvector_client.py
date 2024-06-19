import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection, sql, Cursor

from ..base.base_client import BaseClient, BaseIndexConfig
from .pgvector_config import PgvectorConfig
from ..utility import bytes_to_mb

log = logging.getLogger(__name__)


class PgvectorClient(BaseClient):
    """
    A client for interacting with a PostgreSQL database using the pgvector extension
    (see https://github.com/pgvector/pgvector). Interface is the same as :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig,
                 db_config: PgvectorConfig = PgvectorConfig()) -> None:
        """
        Initialize the PgvectorClient with the specified parameters.

        :param dimension: The dimensionality of the vectors.
        :param index_config: Configuration for the index (see :class:`PgvectorHNSWConfig`
            or :class:`PgvectorIVFFlatConfig`).
        :param db_config: Configuration for the database connection (see :class:`PgvectorConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__table_name = "ecovdbs"
        self.__index_name = "idx:ecovdbs"
        self.__id_name = "id"
        self.__metadata_name = "metadata"
        self.__vector_name = "vector"

        # Establish connection to PostgreSQL database
        self.__conn: Connection = psycopg.connect(
            f"host={db_config.host} port={db_config.port} dbname={db_config.dbname} user={db_config.user} password={db_config.password}")

        # Ensure the vector extension is available
        self.__conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.__conn.commit()

        # Register the vector type for use in PostgreSQL
        register_vector(self.__conn)

        # Drop the existing table and index if they exist
        drop_table = sql.SQL("DROP TABLE IF EXISTS {table_name};").format(table_name=sql.Identifier(self.__table_name))
        self.__conn.execute(drop_table)
        self.__conn.commit()

        drop_index = sql.SQL("DROP INDEX IF EXISTS {index_name};").format(index_name=sql.Identifier(self.__index_name))
        self.__conn.execute(drop_index)
        self.__conn.commit()

        # Create a new table with a vector column of the specified dimension
        create_table = (sql.SQL(
            "CREATE TABLE IF NOT EXISTS {table_name} ({id_name} bigserial PRIMARY KEY, {vector_name} vector({dimension}), {metadata_name} text);").format(
            table_name=sql.Identifier(self.__table_name), id_name=sql.Identifier(self.__id_name),
            vector_name=sql.Identifier(self.__vector_name), dimension=dimension,
            metadata_name=sql.Identifier(self.__metadata_name)))
        self.__conn.execute(create_table)
        self.__conn.commit()
        log.info("Pgvector client initialized")

    def insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        log.info(f"Inserting {len(embeddings)} vectors into database")
        if not metadata or len(metadata) != len(embeddings):
            metadata = ["" for _ in range(len(embeddings))]
        cur: Cursor = self.__conn.cursor()
        with cur.copy(
                sql.SQL(
                    "COPY {table_name} ({id_name}, {vector_name}, {metadata_name}) FROM STDIN (FORMAT BINARY)").format(
                    table_name=sql.Identifier(self.__table_name), id_name=sql.Identifier(self.__id_name),
                    vector_name=sql.Identifier(self.__vector_name),
                    metadata_name=sql.Identifier(self.__metadata_name))) as copy:
            copy.set_types(["bigint", "vector", "text"])
            for i, embedding in enumerate(embeddings):
                copy.write_row((i + start_id, embedding, metadata[i]))
        self.__conn.commit()

    def batch_insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        """
        Not implemented.
        """
        # TODO implement
        pass

    def create_index(self) -> None:
        index_param = self.__index_config.index_param()
        log.info(f"Creating index {self.__index_config.index_param()}")
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
        """
        Set database parameters for index creation and search.

        :param param: A dictionary of parameters to set.
        """
        if param:
            for k, v in param.items():
                command = sql.SQL("SET {key} = {val}").format(key=sql.Identifier(k), val=sql.Literal(v))
                self.__conn.execute(command)
            self.__conn.commit()

    def disk_storage(self) -> float:
        """
        Get the disk storage used by the database. Returns the total on-disk size of the used relation, including data
        and any indexes. It returns the sum of the table size and the index size.

        :return: Disk storage used in MB.
        """
        # https://stackoverflow.com/questions/41991380/whats-the-difference-between-pg-table-size-pg-relation-size-pg-total-relatio/70397779#70397779
        database_size_query = sql.SQL("SELECT pg_total_relation_size({table_name})").format(
            table_name=sql.Literal(self.__table_name))
        res = self.__conn.execute(database_size_query)
        return bytes_to_mb(res.fetchall()[0][0])

    def index_storage(self) -> float:
        database_size_query = sql.SQL("SELECT pg_relation_size({index_name})").format(
            index_name=sql.Literal(self.__index_name))
        res = self.__conn.execute(database_size_query)
        return bytes_to_mb(res.fetchall()[0][0])

    def query(self, query: list[float], k: int) -> list[int]:
        log.info(f"Query {k} vectors. Query: {query}")
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

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        log.info(f"Query {k} vectors with keyword_filter {keyword_filter}. Query: {query}")
        search_param = self.__index_config.search_param()
        self.__set_param(search_param["set"])
        select = sql.Composed([
            sql.SQL(
                "SELECT {id_name} FROM {table_name} WHERE {metadata_name} = {keyword_filter} ORDER BY {vector_name} ").format(
                id_name=sql.Identifier(self.__id_name), table_name=sql.Identifier(self.__table_name),
                metadata_name=sql.Identifier(self.__metadata_name), keyword_filter=sql.Literal(keyword_filter),
                vector_name=sql.Identifier(self.__vector_name)),
            sql.SQL(search_param["metric_operator"]),
            sql.SQL(" %s::vector LIMIT %s::int")
        ])
        # print(self.__conn.execute(sql.SQL("explain analyze ") + select, (query, k)).fetchall()) Post filtern:
        # Reihenfolge der Bearbeitung anders als erwartet. Es werden nicht alle Dateien mit WHERE sortiert und
        # limitiert, sondern alle sortierten limitierten mit where zurückgegeben
        res = self.__conn.execute(select, (query, k))
        return [int(r[0]) for r in res.fetchall()]

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        log.info(f"Query {k} vectors with distance {distance}. Query: {query}")
        search_param = self.__index_config.search_param()
        self.__set_param(search_param["set"])
        select = sql.Composed([
            sql.SQL(
                # TODO umrechnen distanz für andere Metriken außer L2 IP((embedding <#> '[3,1,2]') * -1), Csine (1 -
                #  (embedding <=> '[3,1,2]'))
                "SELECT {id_name} FROM {table_name} WHERE {vector_name} ").format(
                id_name=sql.Identifier(self.__id_name), table_name=sql.Identifier(self.__table_name),
                vector_name=sql.Identifier(self.__vector_name)),
            sql.SQL(search_param["metric_operator"]),
            sql.SQL(" %s::vector < %s::int ORDER BY {vector_name} ").format(
                vector_name=sql.Identifier(self.__vector_name)),
            sql.SQL(search_param["metric_operator"]),
            sql.SQL(" %s::vector LIMIT %s::int")
        ])
        res = self.__conn.execute(select, (query, distance, query, k))
        return [int(r[0]) for r in res.fetchall()]
