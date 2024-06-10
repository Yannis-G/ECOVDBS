from pymilvus import DataType, connections, FieldSchema, CollectionSchema, Collection, utility, SearchResult

from .base_client import BaseClient, BaseIndexConfig
from .milvus_config import MilvusConfig


class MilvusClient(BaseClient):
    """
    A client for interacting with a Milvus database. It extends BaseClient.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: dict | None = None):
        """
        Initialize the MilvusClient with a given database configuration.

        :param dimension: The dimension of the embeddings.
        :param index_config: Configuration for the index.
        :param db_config: Configuration dictionary for the database connection.
        """
        if db_config is None:
            db_config = MilvusConfig().to_dict()
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config
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

    def insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the collection.

        :param embeddings: List of embeddings to insert.
        """
        data = [{"id": i, "vector": v} for i, v in enumerate(embeddings)]
        self.__collection.insert(data=data)
        self.__collection.flush()

    def batch_insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the Milvus database in batches. Not implemented.
        """
        pass

    def crate_index(self) -> None:
        """
        Create an index for the embeddings in the collection.
        """
        index_param: dict = self.__index_config.index_param()
        self.__collection.create_index(self.__vector_name, index_param)

    def disk_storage(self):
        pass

    def index_storage(self):
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the collection with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        search_param: dict = self.__index_config.search_param()
        self.__collection.load()
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name, param=search_param,
                                                     limit=k)
        return [result.id for result in res[0]]
