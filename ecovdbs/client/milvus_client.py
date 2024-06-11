from pymilvus import DataType, connections, FieldSchema, CollectionSchema, Collection, utility, SearchResult

from .base_client import BaseClient, BaseIndexConfig, BaseConfig
from .milvus_config import MilvusConfig, MilvusAutoIndexConfig


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
        self.__vector_name: str = "vector"

        # Connect to the Milvus server
        connections.connect(uri=self.__db_config["uri"])

        # Drop the collection if it already exists
        if utility.has_collection(self.__collection_name):
            utility.drop_collection(self.__collection_name)

        # Define the schema for the collection
        fields: list[FieldSchema] = [
            FieldSchema(name=self.__id_name, dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name=self.__vector_name, dtype=DataType.FLOAT_VECTOR, dim=self.__dimension),
        ]
        schema: CollectionSchema = CollectionSchema(fields)

        # Create the collection with the defined schema
        self.__collection: Collection = Collection(self.__collection_name, schema)

    def insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        data = [{"id": start_id + i, "vector": v} for i, v in enumerate(embeddings)]
        self.__collection.insert(data=data)
        self.__collection.flush()

    def batch_insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        """
        Not implemented.
        """
        pass

    def create_index(self) -> None:
        index_param: dict = self.__index_config.index_param()
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
        search_param: dict = self.__index_config.search_param()
        self.__collection.load()
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name, param=search_param,
                                                     limit=k)
        return [result.id for result in res[0]]
