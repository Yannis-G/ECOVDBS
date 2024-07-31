import logging
from typing import Optional

import docker
from docker.errors import NotFound, APIError
from docker.models.containers import Container
from pymilvus import DataType, connections, FieldSchema, CollectionSchema, Collection, utility, SearchResult

from .milvus_config import MilvusConfig
from ..base_client import BaseClient
from ..base_config import BaseIndexConfig
from ..utility import bytes_to_mb, get_size_of

log = logging.getLogger(__name__)


class MilvusClient(BaseClient):
    """
    A client for interacting with a Milvus database (see https://milvus.io/docs). Interface is the same as
    :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: MilvusConfig = MilvusConfig()):
        """
        Initialize the MilvusClient with a given database configuration.

        :param dimension: The dimension of the embeddings.
        :param index_config: Configuration for the index (see example :class:`MilvusAutoIndexConfig`).
        :param db_config: Configuration for the database connection (see :class:`MilvusConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__collection_name: str = "ecovdbs"
        self.__id_name: str = "id"
        self.__metadata_name: str = "metadata"
        self.__vector_name: str = "vector"
        if db_config.container_name == "milvus-minio":
            self.__persistence_directory: str = "/minio_data/a-bucket/files"
        else:
            self.__persistence_directory: str = "/var/lib/milvus"
            self.__object_storage_directory: str = f"{self.__persistence_directory}/data"

        # Connect to the Milvus server
        connections.connect(uri=db_config.connection_uri)

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

        self.__starting_size: float = 0
        try:
            client = docker.from_env()
            self.__container: Container = client.containers.get(db_config.container_name)
            self.__starting_size: float = self.disk_storage()
            # Delete all files in the persistence directory
            # self.__container.exec_run(f"sh -c 'rm -R -- {self.__persistence_directory}/*'")
        except NotFound | APIError:
            log.error(f"Could not find the database container with the name {db_config.container_name}")

        # Create the collection with the defined schema
        self.__collection: Collection = Collection(self.__collection_name, schema)
        log.info("Milvus client initialized")

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        log.info(f"Inserting {len(embeddings)} vectors into database")
        if not metadata or len(metadata) != len(embeddings):
            metadata = ["" for _ in range(len(embeddings))]

        data = [{self.__id_name: start_id + i,
                 self.__metadata_name: metadata[i],
                 self.__vector_name: v} for i, v in enumerate(embeddings)]
        self.__collection.insert(data=data)
        self.__collection.flush()

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        """
        Not implemented.
        """
        # TODO implement
        pass

    def create_index(self) -> None:
        index_param: dict = self.__index_config.index_param()
        log.info(f"Creating index {self.__index_config.index_param()}")
        self.__collection.create_index(self.__vector_name, index_param)

    def disk_storage(self):
        """
        Get the disk storage used by the database. For a detailed description of the storage see
        https://milvus.io/docs/four_layers.md

        :return: Disk storage used in MB.
        """
        return bytes_to_mb(get_size_of(self.__persistence_directory, self.__container, log)) - self.__starting_size

    def index_storage(self):
        """
        Get the theoretical storage used by the index in the database.

        :return: Theoretical index storage used in MB.
        """
        # Description of the theoretical index storage calculation:
        # https://github.com/milvus-io/milvus/discussions/24894
        nd = self.__collection.num_entities
        m = self.__index_config.index_param()["params"]["M"]
        return bytes_to_mb(nd * self.__dimension * 4 + nd * m * 8)  # 4 bytes per float32, 8 bytes per int64

    def load(self) -> None:
        """
        Load the collection into memory. This is necessary to perform queries on the collection.
        """
        self.__collection.load()

    def query(self, query: list[float], k: int) -> list[int]:
        log.info(f"Query {k} vectors. Query: {query}")
        search_param: dict = self.__index_config.search_param()
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name, param=search_param,
                                                     limit=k)
        return [result.id for result in res[0]]

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        log.info(f"Query {k} vectors with keyword_filter {keyword_filter}. Query: {query}")
        search_param: dict = self.__index_config.search_param()
        expr = f'{self.__metadata_name} == "{keyword_filter}"'
        res: SearchResult = self.__collection.search(data=[query], anns_field=self.__vector_name,
                                                     param=search_param, limit=k, expr=expr)
        return [result.id for result in res[0]]

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        log.info(f"Query {k} vectors with distance {distance}. Query: {query}")
        # TODO implement
        # https://milvus.io/docs/single-vector-search.md#Range-search
