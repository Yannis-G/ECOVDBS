import chromadb
import docker
from chromadb import ClientAPI, Collection, QueryResult
from chromadb.utils.batch_utils import create_batches
from docker.errors import NotFound, APIError

from .base_client import BaseClient, BaseIndexConfig, BaseConfig
from .chroma_config import ChromaConfig, ChromaIndexConfig


class ChromaClient(BaseClient):
    """
    A client for interacting with a Chroma database (see https://docs.trychroma.com/). Interface is the same as
    :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig = ChromaIndexConfig(),
                 db_config: BaseConfig = ChromaConfig()) -> None:
        """
        Initialize the ChromaClient with a given database configuration.

        :param dimension: Not relevant! The first inserted vector decides the dimension of the collection.
        :param index_config: Configuration for the index (see :class:`ChromaIndexConfig`).
        :param db_config: Configuration for the database connection (see :class:`ChromaConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config.to_dict()
        self.__collection_name: str = "ecovdbs"
        self.__persistence_directory = "/chroma/chroma"
        self.__client: ClientAPI = chromadb.HttpClient(host=self.__db_config["host"], port=self.__db_config["port"])
        try:
            client = docker.from_env()
            self.__container = client.containers.get(self.__db_config["container_name"])
            # Delete all subdirectories in the persistence directory
            self.__container.exec_run(f"rm -R -- {self.__persistence_directory}/*/")
        except NotFound | APIError:
            # TODO Errorhandling
            print("No Docker")

        # Ensure the client is alive by checking the heartbeat.
        assert self.__client.heartbeat() is not None

        # Empties and completely resets the database. ⚠️
        self.__client.reset()

        # Get or create the collection.
        self.__collection: Collection = self.__client.get_or_create_collection(name=self.__collection_name,
                                                                               metadata=self.__index_config.index_param())

    def insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        # self.__client.max_batch_size >> 41666
        ids: list[str] = [str(i) for i in range(start_id, start_id + len(embeddings))]
        self.__collection.add(ids=ids, embeddings=embeddings)

    def batch_insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        ids: list[str] = [str(i) for i in range(start_id, start_id + len(embeddings))]
        batches = create_batches(api=self.__client, ids=ids, embeddings=embeddings)
        for batch in batches:
            self.__collection.add(ids=batch[0], embeddings=batch[1])

    def create_index(self):
        """
        Not implemented! Chroma DB automatically creates an index of the embeddings as they are inserted into the
        collection. (https://github.com/zylon-ai/private-gpt/discussions/563)
        """
        pass

    def disk_storage(self):
        """
        Only works if the database runs inside a docker container and the name passed in the config!
        Get the disk storage used by the database. The persistence directory consists of two files: the
        `chroma.sqlite3` and the folder named after the collections UUID for the HNSW index.
        (https://cookbook.chromadb.dev/core/storage-layout/) This method returns the size of the howl persistence
        directory.

        :return: Disk storage used in MB. If the value is negativ the docker container during __init__ was not found.
        """
        return self.__get_size_of(self.__persistence_directory) / 1024 / 1024

    def __get_size_of(self, path: str) -> int:
        """
        Get the size of a directory or file within the Docker container.

        :param path: Path to the directory or file within the container.
        :return: Size in bytes. If the container is not available, return -1.
        """
        # Return -1 if the Docker container is not available
        if not self.__container:
            return -1
        # Execute the 'du' command within the Docker container to get the size of the specified path
        result = self.__container.exec_run(f"du -sb {path}")
        # Check if the command executed successfully
        if result.exit_code == 0:
            output = result.output.decode("utf-8").strip()
            size_in_bytes = int(output.split()[0])
            return size_in_bytes
        else:
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
        return (total_size - sqlite_size) / 1024 / 1024

    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        res: QueryResult = self.__collection.query(query_embeddings=query, n_results=k)
        return [int(id) for id in res["ids"][0]]
