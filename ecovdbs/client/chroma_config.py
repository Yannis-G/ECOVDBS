from .base_config import BaseConfig, BaseIndexConfig, MetricType


class ChromaConfig(BaseConfig):
    """
    Configuration class for Chroma database clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        """
        Initialize the ChromaConfig with default values.

        :param host: The hostname for the database server. Defaults to "localhost".
        :param port: The port number for the database server. Defaults to 8000.
        """
        self.__host: str = host
        self.__port: int = port

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        :return: The configuration as a dictionary.
        """
        return {
            "host": self.__host,
            "port": self.__port
        }


class ChromaIndexConfig(BaseIndexConfig):
    """
    Configuration class for the Chroma HNSW index.
    """
    def __init__(self, metric_type: MetricType = MetricType.L2, construction_ef: int = 100, M: int = 16,
                 search_ef: int = 10, num_threads: int | None = None, resize_factor: float = 1.2, batch_size: int = 100,
                 sync_threshold: int = 1000) -> None:
        """
        Initialize the ChromaIndexConfig for the Chroma HNSW index with default values.

        :param metric_type: Controls the distance metric of the HNSW index. Default: L2
        :param construction_ef: Controls the number of neighbours in the HNSW graph to explore when adding new vectors. The more neighbours HNSW explores the better and more exhaustive the results will be. Increasing the value will also increase memory consumption. Default: 100
        :param M: Controls the maximum number of neighbour connections (M), a newly inserted vector. A higher value results in a mode densely connected graph. The impact on this is slower but more accurate searches with increased memory consumption. Default: 16
        :param search_ef: Controls the number of neighbours in the HNSW graph to explore when searching. Increasing this requires more memory for the HNSW algo to explore the nodes during knn search. Default: 10
        :param num_threads: Controls how many threads HNSW algo use. Default: <number of CPU cores>
        :param resize_factor: Controls the rate of growth of the graph (e.g. how many node capacity will be added) whenever the current graph capacity is reached. Default: 1.2
        :param batch_size: Controls the size of the Bruteforce (in-memory) index. Once this threshold is crossed vectors from BF gets transferred to HNSW index. Default: 100
        :param sync_threshold: Controls the threshold when using HNSW index is written to disk. Default: 1000
        """
        self.__space: MetricType = metric_type
        assert construction_ef > 0, "construction_ef must be positive integer."
        self.__construction_ef: int = construction_ef
        assert M > 0, "M must be positive integer."
        self.__M: int = M
        assert search_ef > 0, "search_ef must be positive integer."
        self.__search_ef: int = search_ef
        assert num_threads is None or num_threads > 0, "num_threads must be positive integer."
        self.__num_threads: int | None = num_threads
        assert resize_factor > 0, "resize_factor must be positive floating point number."
        self.__resize_factor: float = resize_factor
        assert batch_size > 0, "batch_size must be positive integer."
        self.__batch_size: int = batch_size
        assert sync_threshold > 0, "sync_threshold must be positive integer."
        self.__sync_threshold: int = sync_threshold

    def index_param(self) -> dict:
        """
        Generate the index parameters dictionary.

        :return: Dictionary containing the index parameters.
        """
        params: dict = {
            "hnsw:space": self.__space.value.lower(),
            "hnsw:construction_ef": self.__construction_ef,
            "hnsw:M": self.__M,
            "hnsw:search_ef": self.__search_ef,
            "hnsw:resize_factor": self.__resize_factor,
            "hnsw:batch_size": self.__batch_size,
            "hnsw:sync_threshold": self.__sync_threshold
        }
        if self.__num_threads is not None:
            params["hnsw:num_threads"] = self.__num_threads
        return params

    def search_param(self) -> dict:
        """
        No search parameters needed for Chroma DB.

        :return: NotImplementedError
        """
        raise NotImplementedError("No need in Chroma DB, only Params needed for index creation in the collection "
                                  "creation process")
