from abc import ABC, abstractmethod
from enum import Enum


class MetricType(Enum):
    """
    Enum for different types of metrics.
    """
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"


class IndexType(Enum):
    """
    Enum for different types of indexes.
    """
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    IVFSQ8 = "IVF_SQ8"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"
    ES_IVFFlat = "ivfflat"
    GPU_IVF_FLAT = "GPU_IVF_FLAT"
    GPU_IVF_PQ = "GPU_IVF_PQ"
    GPU_CAGRA = "GPU_CAGRA"


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
    def index_param(self) -> dict:
        """
        Get the parameters for the index.

        :return: The index parameters as a dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        """
        Get the parameters for search.

        :return: The search parameters as a dictionary.
        """
        raise NotImplementedError
