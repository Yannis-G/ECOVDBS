from dataclasses import dataclass
from typing import Optional

from ..base_config import BaseConfig, BaseIndexConfig, MetricType, IndexType, BaseHNSWConfig


@dataclass(frozen=True)
class RedisConfig(BaseConfig):
    """
    Configuration class for Redis database clients.

    Attributes:
        host: The hostname for the database server. Defaults to "localhost".
        port: The port number for the database server. Defaults to 6379.
        password: The password for the Redis server. Default to an empty string.
    """
    host: str = 'localhost'
    port: int = 6379
    password: str = ''


class RedisFlatConfig(BaseIndexConfig):
    """
    Configuration class for the Flat index type in Redis.
    """

    def __init__(self, metric_type: MetricType, data_type: str = "FLOAT32", initial_cap: Optional[int] = None,
                 block_size: Optional[int] = None) -> None:
        """
        Initialize the RedisFlatConfig with the specified parameters.

        :param metric_type: The metric type for distance calculation.
        :param data_type: The data type for the vectors. It must be either "FLOAT32" or "FLOAT64". Defaults to
            "FLOAT32".
        :param initial_cap: Initial vector capacity in the index affecting memory allocation size of the index.
        :param block_size: Block size to hold BLOCK_SIZE number of vectors in a contiguous array. This is useful when
            the index is dynamic with respect to addition and deletion.
        """
        assert data_type in ["FLOAT32", "FLOAT64"]
        self.__type = data_type
        self.__index_type: IndexType = IndexType.Flat
        self.__metric_type: MetricType = metric_type
        self.__initial_cap: Optional[int] = initial_cap
        self.__block_size: Optional[int] = block_size

    def index_param(self) -> dict:
        """
        Generate the index parameters dictionary. The directory contains the keys ``index`` for the index and ``param``
        for a directory with params for the index. ``param`` contains the keys ``TYPE`` and ``DISTANCE_METRIC`` and may
        contain the keys ``INITIAL_CAP`` and ``BLOCK_SIZE`` if the value is different from the default value of the
        database.

        :return: A dictionary of index parameters.
        """
        param = {
            "TYPE": self.__type,
            "DISTANCE_METRIC": self.__metric_type.value,
        }
        if self.__initial_cap is not None:
            param["INITIAL_CAP"] = self.__initial_cap
        if self.__block_size is not None:
            param["BLOCK_SIZE"] = self.__block_size
        return {
            "index": self.__index_type.value,
            "param": param
        }

    def search_param(self) -> None:
        """
        No need in Redis DB, only Params needed for index creation in the collection creation process

        :return: None
        """
        return None


class RedisHNSWConfig(BaseHNSWConfig):
    """
    Configuration class for the HNSW index type in Redis.
    """

    def __init__(self, metric_type: MetricType, data_type: str = "FLOAT32", initial_cap: Optional[int] = None,
                 M: Optional[int] = None, ef_construction: Optional[int] = None, ef_runtime: Optional[int] = None,
                 epsilon: Optional[int] = None):
        """
        Initialize the RedisHNSWConfig with the specified parameters.

        :param metric_type: The metric type for distance calculation.
        :param data_type: The data type for the vectors. It must be either "FLOAT32" or "FLOAT64". Defaults to
            "FLOAT32".
        :param initial_cap: Initial vector capacity in the index affecting memory allocation size of the index.
        :param M: Number of maximum allowed outgoing edges for each node in the graph in each layer. On layer zero, the
            maximal number of outgoing edges will be 2M.
        :param ef_construction: Number of maximums allowed potential outgoing edges candidates for each node in the
            graph during the graph building.
        :param ef_runtime: Number of maximum top candidates to hold during the KNN search. Higher values of EF_RUNTIME
            lead to more accurate results at the expense of a longer runtime.
        :param epsilon: Relative factor that sets the boundaries in which a range query may search for candidates. That
            is, vector candidates whose distance from the query vector is radius*(1 + EPSILON) are potentially scanned,
            allowing more extensive search and more accurate results (at the expense of runtime).
        """
        assert data_type in ["FLOAT32", "FLOAT64"]
        self.__type = data_type
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type: MetricType = metric_type
        self.__initial_cap: Optional[int] = initial_cap
        self.__M: Optional[int] = M
        self.__ef_construction: Optional[int] = ef_construction
        self.__ef_runtime: Optional[int] = ef_runtime
        self.__epsilon: Optional[int] = epsilon

    def index_param(self) -> dict:
        """
        Generate the index parameters dictionary. The directory contains the keys ``index`` for the index and ``param``
        for a directory with params for the index. ``param`` contains the keys ``TYPE`` and ``DISTANCE_METRIC`` and may
        contain the keys ``INITIAL_CAP``, ``M``, ``EF_CONSTRUCTION``, ``EF_RUNTIME``, ``EPSILON`` if the value is
        different from the default value of the database.

        :return: A dictionary of index parameters.
        """
        param = {
            "TYPE": self.__type,
            "DISTANCE_METRIC": self.__metric_type.value,
        }
        if self.__initial_cap is not None:
            param["INITIAL_CAP"] = self.__initial_cap
        if self.__M is not None:
            param["M"] = self.__M
        if self.__ef_construction is not None:
            param["EF_CONSTRUCTION"] = self.__ef_construction
        if self.__ef_runtime is not None:
            param["EF_RUNTIME"] = self.__ef_runtime
        if self.__epsilon is not None:
            param["EPSILON"] = self.__epsilon
        return {
            "index": self.__index_type.value,
            "param": param
        }

    def search_param(self) -> Optional[dict]:
        """
        Generate the search parameters dictionary. The directory may contain the key ``EF_RUNTIME`` if the value is
        different from the default value of the database.

        :return: Dictionary containing the search parameters or None if only the default parameters are needed.
        """
        if self.__ef_runtime is not None:
            return {"EF_RUNTIME": self.__ef_runtime}
        return None

    def change_ef_search(self, ef: int) -> None:
        self.__ef_runtime = ef
