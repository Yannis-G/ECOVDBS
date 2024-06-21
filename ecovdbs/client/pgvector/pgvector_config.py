from dataclasses import dataclass
from typing import Optional

from ..base.base_config import BaseConfig, BaseIndexConfig, MetricType, IndexType, BaseHNSWConfig


@dataclass(frozen=True)
class PgvectorConfig(BaseConfig):
    """
    Configuration class for PostgreSQL database clients using the pgvector extension.

    Attributes:
        host: The hostname for the database server. Defaults to "localhost".
        port: The port number for the database server. Defaults to 5432.
        dbname: The database name. Defaults to "postgres".
        user: The username for the database. Defaults to "postgres".
        password: The password for the database. Defaults to "pwd".
    """
    host: str = "localhost"
    port: int = 5432
    dbname: str = "postgres"
    user: str = "postgres"
    password: str = "pwd"


class PgvectorHNSWConfig(BaseHNSWConfig):
    """
    Configuration class for HNSW index in pgvector. An HNSW index creates a multilayer graph. It has better query
    performance than IVFFlat (in terms of speed-recall tradeoff), but has slower build times and uses more memory.
    Also, an index can be created without any data in the table since there isn’t a training step like IVFFlat.
    """

    def __init__(self, metric_type: MetricType, m: Optional[int] = None, ef_construction: Optional[int] = None,
                 ef_search: Optional[int] = None, maintenance_work_mem: Optional[str] = None,
                 max_parallel_maintenance_workers: Optional[int] = None):
        """
        Initialize the PgvectorHNSWConfig with specified parameters.

        :param metric_type: The metric type for distance calculation.
        :param m: The max number of connections per layer.
        :param ef_construction: The size of the dynamic candidate list for constructing the graph. A higher value of
            ef_construction provides better recall at the cost of index build time / insert speed.
        :param ef_search: Specify the size of the dynamic candidate list for search. A higher value provides better
            recall at the cost of speed. This value must be equal or higher than ``k`` in query
        :param maintenance_work_mem: The memory to use for maintenance work. Indexes build significantly faster when
            the graph fits into maintenance_work_mem
        :param max_parallel_maintenance_workers: The maximum number of parallel maintenance workers.
        """
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type = metric_type
        self.__m: Optional[int] = m
        self.__ef_construction: Optional[int] = ef_construction
        self.__ef_search: Optional[int] = ef_search
        self.__maintenance_work_mem: Optional[str] = maintenance_work_mem
        self.__max_parallel_maintenance_workers: Optional[int] = max_parallel_maintenance_workers

    def index_param(self) -> dict:
        """
        Generate the index parameters dictionary. The directory contain the keys ``index_type`` and ``metric_type``
        for the corresponding type. The keys ``with`` and ``set`` contain an empty directory but may contain further
        keys. ``with`` may contain the keys ``m`` and ``ef_construction`` if the value is different from the default
        value of the database. ``set`` may contain the keys ``maintenance_work_mem``
        and ``max_parallel_maintenance_workers``.

        :return: A dictionary of index parameters.
        """
        param = {
            "index_type": self.__index_type.value.lower(),
            "metric_type": _distance_metric[self.__metric_type],
            "with": {},
            "set": {}
        }
        if self.__m is not None:
            param["with"]["m"] = self.__m
        if self.__ef_construction is not None:
            param["with"]["ef_construction"] = self.__ef_construction
        if self.__maintenance_work_mem is not None:
            param["set"]["maintenance_work_mem"] = self.__maintenance_work_mem
        if self.__max_parallel_maintenance_workers is not None:
            param["set"]["max_parallel_maintenance_workers"] = self.__max_parallel_maintenance_workers
        return param

    def search_param(self) -> dict:
        """
        Generate the search parameters dictionary. The directory contain the keys ``metric_operator`` for the operator
        and ``set`` as an empty dictionary. ``set`` may contain the key ``hnsw.ef_search`` if the value is different
        from the default.

        :return: A dictionary of search parameters.
        """
        param = {
            "metric_operator": _metric_operator[self.__metric_type],
            "set": {}
        }
        if self.__ef_search is not None:
            param["set"]["hnsw.ef_search"] = self.__ef_search
        return param

    def change_ef_search(self, ef: int) -> None:
        self.__ef_search = ef


class PgvectorIVFFlatConfig(BaseIndexConfig):
    """
    Configuration class for IVF-Flat index in pgvector. An IVFFlat index divides vectors into lists, and then searches a
    subset of those lists that are closest to the query vector. It has faster build times and uses less memory than
    HNSW, but has lower query performance (in terms of speed-recall tradeoff).

    Three keys to achieving good recall are:
        - Create the index after the table has some data
        - Choose an appropriate number of lists - a good place to start is ``rows / 1000`` for up to 1M rows and ``sqrt(rows)`` for over 1M rows
        - When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) - a good place to start is ``sqrt(lists)``
    """

    def __init__(self, metric_type: MetricType, lists: int, probes: Optional[int] = None,
                 max_parallel_maintenance_workers: Optional[int] = None):
        """
        Initialize the PgvectorIVFFlatConfig with specified parameters. For more details
        see :class:`PgvectorIVFFlatConfig`.

        :param metric_type: The metric type for distance calculation.
        :param lists: The number of lists for the IVF-Flat index.
        :param probes: The number of probes for the IVF-Flat index.
        :param max_parallel_maintenance_workers: The maximum number of parallel maintenance workers.
        """
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type = metric_type
        self.__lists: int = lists
        self.__probes: Optional[int] = probes
        self.__max_parallel_maintenance_workers: Optional[int] = max_parallel_maintenance_workers

    def index_param(self) -> dict:
        """
        Generate the index parameters dictionary. The directory contain the keys ``index_type`` and ``metric_type``
        for the corresponding type. The key ``with`` contains a directory with the key ``lists``. The key ``set``
        contain an empty directory but may contain the key ``max_parallel_maintenance_workers`` if the value is
        different from the default value of the database.

        :return: A dictionary of index parameters.
        """
        param = {
            "index_type": self.__index_type.value.lower().replace("_", ""),
            "metric_type": _distance_metric[self.__metric_type],
            "with": {
                "lists": self.__lists
            },
            "set": {}
        }
        if self.__max_parallel_maintenance_workers is not None:
            param["set"]["max_parallel_maintenance_workers"] = self.__max_parallel_maintenance_workers
        return param

    def search_param(self) -> dict:
        """
        Generate the search parameters dictionary. The directory contain the keys ``metric_operator`` for the operator
        and ``set`` as an empty dictionary. ``set`` may contain the key ``ivfflat.probes`` if the value is different
        from the default.

        :return: A dictionary of search parameters.
        """
        param = {
            "metric_operator": _metric_operator[self.__metric_type],
            "set": {}
        }
        if self.__probes is not None:
            param["set"]["ivfflat.probes"] = self.__probes
        return param


# Mapping of MetricType to pgvector distance metric
_distance_metric = {
    MetricType.L2: "vector_l2_ops",
    MetricType.IP: "vector_ip_ops",
    MetricType.COSINE: "vector_cosine_ops"
}

# Mapping of MetricType to pgvector metric operator
_metric_operator = {
    MetricType.L2: "<->",
    MetricType.IP: "<#>",
    MetricType.COSINE: "<=>"
}
