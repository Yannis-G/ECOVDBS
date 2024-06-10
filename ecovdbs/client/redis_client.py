import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from .base_client import BaseClient
from .base_config import BaseIndexConfig
from .redis_config import RedisConfig


class RedisClient(BaseClient):
    """
    A client for interacting with a Redis database using vector embeddings. It extends BaseClient.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig | None, db_config: dict | None = None) -> None:
        """
        Initialize the RedisClient with given database and index configurations.

        :param dimension: The dimension of the vector embeddings.
        :param index_config: Configuration for the index.
        :param db_config: Configuration dictionary for the database connection.
        """
        if db_config is None:
            db_config = RedisConfig().to_dict()

        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config
        self.__index_name: str = "ecovdbs"
        self.__vector_name: str = "vector"

        # Initialize the Redis client
        self.__client: Redis = Redis(host=self.__db_config["host"], port=self.__db_config["port"],
                                     password=self.__db_config["password"])

        # Flush the database to ensure it's empty
        self.__client.flushdb()

    def insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the Redis database.

        :param embeddings: List of embeddings to insert.
        """
        pipeline = self.__client.pipeline()
        for i, embedding in enumerate(embeddings):
            pipeline.json().set(i, "$", {self.__vector_name: embedding})
        pipeline.execute()

    def batch_insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the Redis database in batches. Not implemented.
        """
        pass

    def crate_index(self) -> None:
        """
        Create an index for the embeddings in the Redis database.
        """
        param = self.__index_config.index_param()
        param["param"]["DIM"] = self.__dimension
        fields = [
            VectorField(name=f"$.{self.__vector_name}", algorithm=param["index"], attributes=param["param"],
                        as_name=self.__vector_name),
        ]
        definition = IndexDefinition(index_type=IndexType.JSON)
        self.__client.ft(self.__index_name).create_index(fields=fields, definition=definition)

    def disk_storage(self):
        """
        Get the disk storage used by the Redis database.

        :return: Disk storage used in MB.
        """
        return self.__client.info("memory")["used_memory_dataset"] / 1024 / 1024

    def index_storage(self):
        """
        Get the storage used by the index in the Redis database.

        :return: Index storage used in MB.
        """
        return self.__client.ft(self.__index_name).info()["vector_index_sz_mb"]

    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the Redis database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        redis_query = Query(f"(*)=>[KNN {k} @{self.__vector_name} $query_vector AS vector_score]").sort_by(
            "vector_score").return_fields("vector_score", "id").paging(0, k).dialect(2)
        res = self.__client.ft(self.__index_name).search(redis_query, {
            "query_vector": np.array(query, dtype=np.float32).tobytes()}).docs
        return [int(doc['id']) for doc in res]
