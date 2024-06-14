from abc import abstractmethod, ABC
from .base_config import BaseIndexConfig, BaseConfig


class BaseClient(ABC):
    """
    Abstract base class defining the interface for database clients.
    """

    @abstractmethod
    def __init__(self, dimension: int, index_config: BaseIndexConfig, db_config: BaseConfig) -> None:
        """
        Initialize the database client with the given configuration.

        :param dimension: The dimensionality of the embeddings.
        :param index_config: Configuration for the index.
        :param db_config: Configuration for the database connection.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        """
        Insert embeddings into the database. The ids for the vector a starting with ``start_id`` and a counted
        upwards.

        :param embeddings: List of embeddings to insert.
        :param start_id: Index of the first inserted vector.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        """
        Insert embeddings into the database in batches to improve efficiency. The ids for the vector a starting with
        ``start_id`` and a counted upwards.

        :param embeddings: List of embeddings to insert.
        :param start_id: Index of the first inserted vector.
        """
        raise NotImplementedError

    @abstractmethod
    def create_index(self) -> None:
        """
        Create an index in the database. For details see the documentation of the given index configuration in __init__.
        """
        raise NotImplementedError

    @abstractmethod
    def disk_storage(self):
        """
        Get the disk storage used by the database.

        :return: Disk storage used in MB.
        """
        raise NotImplementedError

    @abstractmethod
    def index_storage(self):
        """
        Get the storage used by the index in the database.

        :return: Index storage used in MB.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the database with a given embedding and return the top k results. For details of the search parameters see
        the documentation of the given index configuration in __init__.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        raise NotImplementedError
