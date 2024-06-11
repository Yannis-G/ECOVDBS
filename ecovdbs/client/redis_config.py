from .base_config import BaseConfig, BaseIndexConfig, MetricType, IndexType


class RedisConfig(BaseConfig):
    """
    Configuration class for Redis database clients.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, password="") -> None:
        """
        Initialize the RedisConfig with default values.

        :param host: The hostname for the database server. Defaults to "localhost".
        :param port: The port number for the database server. Defaults to 6379.
        :param password: The password for the Redis server. Defaults to an empty string.
        """
        self.__host: str = host
        self.__port: int = port
        self.__password: str = password

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary. The dictionary will have the keys host for the ``hostname`` of the
        database server, ``port`` for th port for the database server and ``password`` for the password

        :return: The configuration as a dictionary.
        """
        return {
            "host": self.__host,
            "port": self.__port,
            "password": self.__password
        }


class RedisFlatConfig(BaseIndexConfig):
    """
    Configuration class for the Flat index type in Redis.
    """

    def __init__(self, metric_type: MetricType, data_type: str = "FLOAT32", initial_cap: int | None = None,
                 block_size: int | None = None) -> None:
        """
        Initialize the RedisFlatConfig with the specified parameters.

        :param metric_type: The metric type for distance calculation.
        :param data_type: The data type for the vectors. Must be either "FLOAT32" or "FLOAT64". Defaults to "FLOAT32".
        :param initial_cap: Initial vector capacity in the index affecting memory allocation size of the index.
        :param block_size: Block size to hold BLOCK_SIZE amount of vectors in a contiguous array. This is useful when
            the index is dynamic with respect to addition and deletion.
        """
        assert data_type in ["FLOAT32", "FLOAT64"]
        self.__type = data_type
        self.__index_type: IndexType = IndexType.Flat
        self.__metric_type: MetricType = metric_type
        self.__initial_cap: int | None = initial_cap
        self.__block_size: int | None = block_size

    def index_param(self) -> dict | None:
        """
        Generate the index parameters dictionary. The directory contain the keys ``index`` for the index and ``param``
        for a  directory with params for the index. ``param`` contains the keys``TYPE`` and ``DISTANCE_METRIC`` and may
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

    def search_param(self) -> dict | None:
        """
        No need in Redis DB, only Params needed for index creation in the collection creation process

        :return: None
        """
        return None


class RedisHNSWConfig(BaseIndexConfig):
    """
    Configuration class for the HNSW index type in Redis.
    """

    def __init__(self, metric_type: MetricType, data_type: str = "FLOAT32", initial_cap: int | None = None,
                 M: int | None = None, ef_construction: int | None = None, ef_runtime: int | None = None,
                 epsilon: int | None = None):
        """
        Initialize the RedisHNSWConfig with the specified parameters.

        :param metric_type: The metric type for distance calculation.
        :param data_type: The data type for the vectors. Must be either "FLOAT32" or "FLOAT64". Defaults to "FLOAT32".
        :param initial_cap: Initial vector capacity in the index affecting memory allocation size of the index.
        :param M: Number of maximum allowed outgoing edges for each node in the graph in each layer. on layer zero the
            maximal number of outgoing edges will be 2M..
        :param ef_construction: Number of maximum allowed potential outgoing edges candidates for each node in the
            graph, during the graph building.
        :param ef_runtime: Number of maximum top candidates to hold during the KNN search. Higher values of EF_RUNTIME
            lead to more accurate results at the expense of a longer runtime.
        :param epsilon: Relative factor that sets the boundaries in which a range query may search for candidates. That
            is, vector candidates whose distance from the query vector is radius*(1 + EPSILON) are potentially scanned,
            allowing more extensive search and more accurate results (on the expense of runtime).
        """
        assert data_type in ["FLOAT32", "FLOAT64"]
        self.__type = data_type
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type: MetricType = metric_type
        self.__initial_cap: int | None = initial_cap
        self.__M: int | None = M
        self.__ef_construction: int | None = ef_construction
        self.__ef_runtime: int | None = ef_runtime
        self.__epsilon: int | None = epsilon

    def index_param(self) -> dict | None:
        """
        Generate the index parameters dictionary. The directory contain the keys ``index`` for the index and ``param``
        for a  directory with params for the index. ``param`` contains the keys``TYPE`` and ``DISTANCE_METRIC`` and may
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

    def search_param(self) -> dict | None:
        """
        No need in Redis DB, only Params needed for index creation in the collection creation process

        :return: None
        """
        return None
