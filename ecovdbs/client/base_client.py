from abc import abstractmethod, ABC


class BaseClient(ABC):
    """
    Abstract base class defining the interface for database clients.
    """

    @abstractmethod
    def __init__(self, dimension: int, db_config: dict) -> None:
        """
        Initialize the database client with the given configuration.

        :param db_config: Configuration dictionary for the database connection.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, embeddings: list[list[float]]) -> None:
        """
        Insert embeddings into the database.

        :param embeddings: List of embeddings to insert.
        :return:
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
        raise NotImplementedError

    @abstractmethod
    def index_storage(self):
        raise NotImplementedError

    @abstractmethod
    def query(self, query: list[float], k: int) -> list[list[float]]:
        """
        Query the database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The top k results from the query.
        """
        raise NotImplementedError
