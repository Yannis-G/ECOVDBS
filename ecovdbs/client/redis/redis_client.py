import logging
from typing import Optional

import numpy as np
from redis import Redis
from redis.client import Pipeline
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .redis_config import RedisConfig
from ..base_client import BaseClient
from ..base_config import BaseIndexConfig
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
        self.__batch_size = 1000
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

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        if len(embeddings) > self.__batch_size:
            self.batch_insert(embeddings, metadata, start_id)
        else:
            pipeline, metadata = self.__pre_insert(len(embeddings), metadata)
            for i, embedding in enumerate(embeddings):
                pipeline.hset(str(i + start_id),
                              mapping={self.__vector_name: np.array(embedding).astype(self.__vector_dtype).tobytes(),
                                       self.__metadata_name: metadata[i]})
            pipeline.execute()

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        pipeline, metadata = self.__pre_insert(len(embeddings), metadata)
        for i, embedding in enumerate(embeddings):
            pipeline.hset(str(i + start_id),
                          mapping={self.__vector_name: np.array(embedding).astype(self.__vector_dtype).tobytes(),
                                   self.__metadata_name: metadata[i]})
            if i % self.__batch_size == 0:
                pipeline.execute()
        pipeline.execute()

    def __pre_insert(self, len_embeddings: int, metadata: Optional[list[str]]) -> tuple[Pipeline, list[str]]:
        """
        Prepares a pipeline for inserting embeddings into the database.

        :param len_embeddings: The number of embeddings to insert.
        :param metadata: The metadata for the embeddings.
        :return: A pipeline for inserting the embeddings and the metadata.
        """
        log.info(f"Inserting {len_embeddings} vectors into database")
        pipeline = self.__client.pipeline()
        if not metadata or len(metadata) != len_embeddings:
            metadata = ["" for _ in range(len_embeddings)]
        return pipeline, metadata

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
        """
        Prepares a string representation of the EF_RUNTIME parameter for the query.

        If the search parameter (`param`) exists in the index configuration, it returns a formatted string with the
        EF_RUNTIME value. Otherwise, it returns an empty string.

        :return: Formatted string with EF_RUNTIME parameter or an empty string.
        """
        param = self.__index_config.search_param()
        return f"$EF_RUNTIME: {param['EF_RUNTIME']}; " if param else ""

    def load(self) -> None:
        """
        Not implemented. No need for Redis.
        """
        return None

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
