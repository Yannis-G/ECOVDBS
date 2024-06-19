import logging

import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from ..base.base_client import BaseClient, BaseIndexConfig
from .redis_config import RedisConfig
from ..utility import bytes_to_mb

log = logging.getLogger(__name__)


class RedisClient(BaseClient):
    """
    A client for interacting with a Redis database using vector embeddings
    (see https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/). Interface is the
    same as :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: RedisConfig = RedisConfig()) -> None:
        """
        Initialize the RedisClient with given database and index configurations.

        :param dimension: The dimension of the vector embeddings.
        :param index_config: Configuration for the index (see :class:`RedisFlatConfig` or :class:`RedisHNSWConfig`).
        :param db_config: Configuration for the database connection (see :class:`RedisConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__index_name: str = "ecovdbs"
        self.__metadata_name: str = "metadata"
        self.__vector_name: str = "vector"
        if self.__index_config.index_param()["param"]["TYPE"] == "FLOAT32":
            self.__vector_dtype = np.float32
        else:
            self.__vector_dtype = np.float64

        # Initialize the Redis client
        self.__client: Redis = Redis(host=db_config.host, port=db_config.port, password=db_config.password)

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
        # TODO implement
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
        return round(float(self.__client.ft(self.__index_name).info()["vector_index_sz_mb"]), 2)

    def __pre_query(self) -> str:
        param = self.__index_config.search_param()
        return f"$EF_RUNTIME: {param['EF_RUNTIME']}; " if param else ""

    def query(self, query: list[float], k: int) -> list[int]:
        log.info(f"Query {k} vectors. Query: {query}")
        redis_query = Query(
            f"(*)=>[KNN {k} @{self.__vector_name} $query_vector]=>{{{self.__pre_query()}$YIELD_DISTANCE_AS: vector_score}}").sort_by(
            "vector_score").return_fields("vector_score", "id", "metadata").paging(0, k).dialect(2)
        res = self.__client.ft(self.__index_name).search(redis_query, {
            "query_vector": np.array(query, dtype=np.float32).tobytes()}).docs
        return [int(doc['id']) for doc in res]

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        log.info(f"Query {k} vectors with keyword_filter {keyword_filter}. Query: {query}")
        redis_query = Query(
            f"(@{self.__metadata_name}:{keyword_filter})=>[KNN {k} @{self.__vector_name} $query_vector]=>{{{self.__pre_query()}$YIELD_DISTANCE_AS: vector_score}}").sort_by(
            "vector_score").return_fields("vector_score", "id", "metadata").paging(0, k).dialect(2)
        res = self.__client.ft(self.__index_name).search(redis_query, {
            "query_vector": np.array(query, dtype=np.float32).tobytes()}).docs
        return [int(doc['id']) for doc in res]

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        log.info(f"Query {k} vectors with distance {distance}. Query: {query}")
        redis_query = Query(f"@{self.__vector_name}: [VECTOR_RANGE {distance} $query_vector]=>{{$YIELD_DISTANCE_AS: "
                            f"vector_score}}").sort_by(
            "vector_score").return_fields("vector_score", "id", "metadata").paging(0, k).dialect(2)
        res = self.__client.ft(self.__index_name).search(redis_query, {
            "query_vector": np.array(query, dtype=np.float32).tobytes()}).docs
        return [int(doc['id']) for doc in res]
