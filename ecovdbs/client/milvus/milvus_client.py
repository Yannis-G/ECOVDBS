import logging

from pymilvus import DataType, connections, FieldSchema, CollectionSchema, Collection, utility, SearchResult

from ..base.base_client import BaseClient, BaseIndexConfig, BaseConfig
from .milvus_config import MilvusConfig

log = logging.getLogger(__name__)


class MilvusClient(BaseClient):
    """
    A client for interacting with a Milvus database (see https://milvus.io/docs). Interface is the same as
    :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: BaseConfig = MilvusConfig()):
        """
        Initialize the MilvusClient with a given database configuration.

        :param dimension: The dimension of the embeddings.
        :param index_config: Configuration for the index (see example :class:`MilvusAutoIndexConfig`).
        :param db_config: Configuration for the database connection (see :class:`MilvusConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config.to_dict()
        self.__collection_name: str = "ecovdbs"
        self.__id_name: str = "id"
        self.__metadata_name: str = "metadata"
        self.__vector_name: str = "vector"

        # Connect to the Milvus server
        connections.connect(uri=self.__db_config["uri"])

        # Drop the collection if it already exists
        if utility.has_collection(self.__collection_name):
            utility.drop_collection(self.__collection_name)

        # Define the schema for the collection
        fields: list[FieldSchema] = [
            FieldSchema(name=self.__id_name, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=self.__metadata_name, dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name=self.__vector_name, dtype=DataType.FLOAT_VECTOR, dim=self.__dimension),
        ]
        schema: CollectionSchema = CollectionSchema(fields)

        # Create the collection with the defined schema
        self.__collection: Collection = Collection(self.__collection_name, schema)
        log.info("Milvus client initialized")

    def insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        log.info(f"Inserting {len(embeddings)} vectors into database")
        if not metadata or len(metadata) != len(embeddings):
            metadata = ["" for _ in range(len(embeddings))]

        data = [{self.__id_name: start_id + i,
                 self.__metadata_name: metadata[i],
                 self.__vector_name: v} for i, v in enumerate(embeddings)]
        self.__collection.insert(data=data)
        self.__collection.flush()

    def batch_insert(self, embeddings: list[list[float]], metadata: list[str] | None = None, start_id: int = 0) -> None:
        """
        Not implemented.
        """
        pass

    def create_index(self) -> None:
        index_param: dict = self.__index_config.index_param()
        log.info(f"Creating index {self.__index_config.index_param()}")
        self.__collection.create_index(self.__vector_name, index_param)

    def disk_storage(self):
        """
        Not implemented.
        """
        pass

    def index_storage(self):
        """
        Not implemented.
        """
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        log.info(f"Query {k} vectors. Query: {query}")
        search_param: dict = self.__index_config.search_param()
        self.__collection.load()
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name, param=search_param,
                                                     limit=k)
        return [result.id for result in res[0]]

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        log.info(f"Query {k} vectors with keyword_filter {keyword_filter}. Query: {query}")
        search_param: dict = self.__index_config.search_param()
        self.__collection.load()
        expr = f'{self.__metadata_name} == "{keyword_filter}"'
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name,
                                                     param=search_param, limit=k, expr=expr)
        return [result.id for result in res[0]]

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        log.info(f"Query {k} vectors with distance {distance}. Query: {query}")
