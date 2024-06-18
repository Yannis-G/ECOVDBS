from abc import ABC, abstractmethod
from enum import Enum


class MetricType(Enum):
    """
    Enum for different types of metrics.
    L2 = Euclidean Distance
    IP = Inner Product
    COSINE = Cosine similarity
    """
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"


class IndexType(Enum):
    """
    Enum for different types of indexes.
    """
    Flat = "FLAT"
    IVFFlat = "IVF_FLAT"
    IVFSQ8 = "IVF_SQ8"
    IVFPQ = "IVF_PQ"
    HNSW = "HNSW"
    SCANN = "SCANN"
    DISKANN = "DISKANN"
    AUTOINDEX = "AUTOINDEX"


class BaseConfig(ABC):
    """
    Abstract base class defining the configuration interface for database clients.
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        :return: The configuration as a dictionary.
        """
        raise NotImplementedError


class BaseIndexConfig(ABC):
    """
    Abstract base class defining the configuration interface for index settings.
    """

    @abstractmethod
    def index_param(self) -> dict | None:
        """
        Get the parameters for the index.

        :return: The index parameters as a dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict | None:
        """
        Get the parameters for search.

        :return: The search parameters as a dictionary.
        """
        raise NotImplementedError


class BaseHNSWConfig(BaseIndexConfig, ABC):
    """
    Abstract base class defining the configuration interface for HNSW settings.
    """

    @abstractmethod
    def change_ef_search(self, ef: int) -> None:
        """
        Change the EF of the HNSW.

        :param ef: Parameter controlling query time/accuracy trade-off. Higher ef leads to more accurate but slower
            search.
        """
        raise NotImplementedError
