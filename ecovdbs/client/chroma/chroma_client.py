import logging
from typing import Optional

import chromadb
import docker
from chromadb import ClientAPI, Collection, QueryResult
from chromadb.utils.batch_utils import create_batches
from docker.errors import NotFound, APIError

from .chroma_config import ChromaConfig, ChromaHNSWConfig
from ..base_client import BaseClient
from ..base_config import BaseIndexConfig
from ..utility import bytes_to_mb

log = logging.getLogger(__name__)


class ChromaClient(BaseClient):
    """
    A client for interacting with a Chroma database (see https://docs.trychroma.com/). Interface is the same as
    :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig = ChromaHNSWConfig(),
                 db_config: ChromaConfig = ChromaConfig()) -> None:
        """
        Initialize the ChromaClient with a given database configuration.

        :param dimension: Not relevant! The first inserted vector decides the dimension of the collection.
        :param index_config: Configuration for the index (see :class:`ChromaIndexConfig`).
        :param db_config: Configuration for the database connection (see :class:`ChromaConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__search_param = self.__index_config.search_param()
        self.__collection_name: str = "ecovdbs"
        self.__metadata_field: str = "metadata"
        self.__persistence_directory = "/chroma/chroma"
        self.__client: ClientAPI = chromadb.HttpClient(host=db_config.host, port=db_config.port)

        # Ensure the client is alive by checking the heartbeat.
        assert self.__client.heartbeat() is not None

        try:
            client = docker.from_env()
            self.__container = client.containers.get(db_config.container_name)
            # Delete all subdirectories in the persistence directory
            self.__container.exec_run(f"sh -c 'rm -R -- {self.__persistence_directory}/*/'")
        except NotFound | APIError:
            log.error(f"Could not find the database container with the name {db_config.container_name}")

        # Empties and completely resets the database. ⚠️
        self.__client.reset()

        # Get or create the collection.
        self.__collection: Collection = self.__client.get_or_create_collection(name=self.__collection_name,
                                                                               metadata=self.__index_config.index_param())
        log.info("Chroma client initialized")

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        ids, metadata = self.__pre_insert(len(embeddings), metadata, start_id)
        # self.__client.max_batch_size >> 41666
        self.__collection.add(ids=ids, embeddings=embeddings, metadatas=metadata)

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        ids, metadata = self.__pre_insert(len(embeddings), metadata, start_id)
        batches = create_batches(api=self.__client, ids=ids, embeddings=embeddings, metadatas=metadata)
        for batch in batches:
            self.__collection.add(ids=batch[0], embeddings=batch[1], metadatas=batch[2])

    def __pre_insert(self, len_embeddings: int, metadata: Optional[list[str]], start_id: int):
        """
        Prepare data for insertion into the database.

        This method generates a list of IDs for the embeddings and formats the metadata if provided.

        :param len_embeddings: Number of embeddings to insert.
        :param metadata: List of metadata strings to insert. The length should match len_embeddings if provided.
        :param start_id: Index of the first inserted vector.
        :return: A tuple containing the list of IDs and formatted metadata.
        """
        log.info(f"Inserting {len_embeddings} vectors into database")
        ids: list[str] = [str(i) for i in range(start_id, start_id + len_embeddings)]
        metadata = [{self.__metadata_field: md} for md in metadata] if metadata and len(
            metadata) == len_embeddings else None
        return ids, metadata

    def create_index(self) -> float:
        """
        Not implemented! Chroma DB automatically creates an index of the embeddings as they are inserted into the
        collection. (https://github.com/zylon-ai/private-gpt/discussions/563)
        """
        pass

    def disk_storage(self) -> float:
        """
        Only works if the database runs inside a docker container and the name passed in the config!
        Get the disk storage used by the database. The persistence directory consists of two files: the
        `chroma.sqlite3` and the folder named after the collections UUID for the HNSW index.
        (https://cookbook.chromadb.dev/core/storage-layout/) This method returns the size of the howl persistence
        directory.

        :return: Disk storage used in MB. If the value is negativ the docker container during __init__ was not found.
        """
        return bytes_to_mb(self.__get_size_of(self.__persistence_directory))

    def __get_size_of(self, path: str) -> int:
        """
        Get the size of a directory or file within the Docker container.

        :param path: Path to the directory or file within the container.
        :return: Size in bytes. If the container is not available, return -1.
        """
        # Return -1 if the Docker container is not available
        if not self.__container:
            log.error("The database container was not found.")
            return -1
        # Execute the 'du' command within the Docker container to get the size of the specified path
        result = self.__container.exec_run(f"du -sb {path}")
        # Check if the command executed successfully
        if result.exit_code == 0:
            output = result.output.decode("utf-8").strip()
            size_in_bytes = int(output.split()[0])
            return size_in_bytes
        else:
            log.error(f"The database container returned an error: {result.exit_code}")
            return -1

    def index_storage(self):
        """
        Only works if the database runs inside a docker container and the name passed in the config!
        Get the storage used by the index in the database. The persistence directory consists of two files: the
        `chroma.sqlite3` and the folder named after the collections UUID for the HNSW index.
        (https://cookbook.chromadb.dev/core/storage-layout/) This method returns the size of thd folder for the HNSW
        index.

        :return: Index storage used in MB. If the value is negativ the docker container during __init__ was not found.
        """
        total_size = self.__get_size_of(self.__persistence_directory)
        sqlite_size = self.__get_size_of(f"{self.__persistence_directory}/chroma.sqlite3")
        return bytes_to_mb(total_size - sqlite_size)

    def __pre_query(self) -> None:
        """
        Update collections metadata.
        """
        search_param = self.__index_config.search_param()
        if search_param != self.__search_param:
            self.__search_param = search_param
            self.__collection: Collection = self.__client.get_or_create_collection(name=self.__collection_name,
                                                                                   metadata=self.__index_config.search_param())

    #
    def query(self, query: list[float], k: int) -> list[int]:
        log.info(f"Query {k} vectors. Query: {query}")
        self.__pre_query()
        res: QueryResult = self.__collection.query(query_embeddings=query, n_results=k)
        return [int(id) for id in res["ids"][0]]

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        log.info(f"Query {k} vectors with keyword_filter {keyword_filter}. Query: {query}")
        self.__pre_query()
        res: QueryResult = self.__collection.query(query_embeddings=query, n_results=k,
                                                   where={self.__metadata_field: keyword_filter})
        return [int(id) for id in res["ids"][0]]

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        """
        Chroma DB does not support ranged queries.
        """
        raise NotImplementedError
