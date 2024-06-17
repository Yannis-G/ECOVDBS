import logging

import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..base.base_client import BaseClient, BaseConfig, BaseIndexConfig
from .redis_config import RedisConfig
from ..utility import bytes_to_mb

log = logging.getLogger(__name__)


class RedisClient(BaseClient):
    """
    A client for interacting with a Redis database using vector embeddings
    (see https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/). Interface is the
    same as :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: BaseConfig = RedisConfig()) -> None:
        """
        Initialize the RedisClient with given database and index configurations.

        :param dimension: The dimension of the vector embeddings.
        :param index_config: Configuration for the index (see :class:`RedisIndexConfig` or :class:`RedisHNSWConfig`).
        :param db_config: Configuration for the database connection (see :class:`RedisConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config.to_dict()
        self.__index_name: str = "ecovdbs"
        self.__metadata_name: str = "metadata"
        self.__vector_name: str = "vector"
        if self.__index_config.index_param()["param"]["TYPE"] == "FLOAT32":
            self.__vector_dtype = np.float32
        else:
            self.__vector_dtype = np.float64

        # Initialize the Redis client
        self.__client: Redis = Redis(host=self.__db_config["host"], port=self.__db_config["port"],
                                     password=self.__db_config["password"])

        # Flush the database to ensure it's empty
        self.__client.flushdb()
        log.info("Redis client initialized")

    def insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        log.info(f"Inserting {len(embeddings)} vectors into database")
        pipeline = self.__client.pipeline()
        if not metadata or len(metadata) != len(embeddings):
            metadata = ["" for _ in range(len(embeddings))]

        for i, embedding in enumerate(embeddings):
            pipeline.hset(str(i + start_id),
                          mapping={self.__vector_name: np.array(embedding).astype(self.__vector_dtype).tobytes(),
                                   self.__metadata_name: metadata[i]})
        pipeline.execute()

    def batch_insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        """
        Not implemented.
        """
        pass

    def create_index(self) -> None:
        param = self.__index_config.index_param()
        log.info(f"Creating index {self.__index_config.index_param()}")
        param["param"]["DIM"] = self.__dimension
        fields = [
            VectorField(name=self.__vector_name, algorithm=param["index"], attributes=param["param"],
                        as_name=self.__vector_name),
            TextField(self.__metadata_name)
        ]
        definition = IndexDefinition(index_type=IndexType.HASH)
        self.__client.ft(self.__index_name).create_index(fields=fields, definition=definition)

    def disk_storage(self) -> float:
        # The size in bytes of the dataset (used_memory_overhead subtracted from used_memory)
        return bytes_to_mb(self.__client.info("memory")["used_memory_dataset"])

    def index_storage(self) -> float:
        return self.__client.ft(self.__index_name).info()["vector_index_sz_mb"]

    def query(self, query: list[float], k: int, keyword_filter: str | None = None) -> list[int]:
        """
        Query the database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :param keyword_filter: A keyword-based filter to restrict the results. The metadate field of the result is equal
            to keyword_filter
        :return: The id of the top k results from the query.
        """
        log.info(f"Query {k} vectors with {keyword_filter}. Query: {query}")
        pre_filter = f"@{self.__metadata_name}:{keyword_filter}" if keyword_filter else "*"
        redis_query = Query(f"({pre_filter})=>[KNN {k} @{self.__vector_name} $query_vector AS vector_score]").sort_by(
            "vector_score").return_fields("vector_score", "id", "metadata").paging(0, k).dialect(2)
        res = self.__client.ft(self.__index_name).search(redis_query, {
            "query_vector": np.array(query, dtype=np.float32).tobytes()}).docs
        return [int(doc['id']) for doc in res]
