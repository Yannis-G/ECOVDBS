from abc import abstractmethod, ABC
from .base_config import BaseIndexConfig


class BaseClient(ABC):
    """
    Abstract base class defining the interface for database clients.
    """

    @abstractmethod
    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: dict) -> None:
        """
        Initialize the database client with the given configuration.

        :param dimension: The dimensionality of the embeddings.
        :param index_config: Configuration for the index.
        :param db_config: Configuration dictionary for the database connection.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the database.

        :param embeddings: List of embeddings to insert.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the database in batches.

        :param embeddings: List of embeddings to insert.
        """
        raise NotImplementedError

    @abstractmethod
    def crate_index(self) -> None:
        """
        Create an index in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def disk_storage(self):
        """
        Get the disk storage used by the Redis database.

        :return: Disk storage used in MB.
        """
        raise NotImplementedError

    @abstractmethod
    def index_storage(self):
        """
        Get the storage used by the index in the Redis database.

        :return: Index storage used in MB.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        raise NotImplementedError
