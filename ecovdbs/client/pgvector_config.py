from .base_config import BaseConfig, BaseIndexConfig, MetricType, IndexType


class PgvectorConfig(BaseConfig):
    def __init__(self, host: str = "localhost", port: int = 5432, dbname: str = "postgres", user: str = "postgres",
                 password: str = "pwd"):
        self.__host = host
        self.__port = port
        self.__dbname = dbname
        self.__user = user
        self.__password = password

    def to_dict(self) -> dict:
        return {
            "host": self.__host,
            "port": self.__port,
            "dbname": self.__dbname,
            "user": self.__user,
            "password": self.__password
        }


class PgvectorHNSWConfig(BaseIndexConfig):
    def __init__(self, metric_type: MetricType, m: int | None = None, ef_construction: int | None = None,
                 ef_search: int | None = None, maintenance_work_mem: str | None = None,
                 max_parallel_maintenance_workers: int | None = None):
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type = metric_type
        self.__m: int | None = m
        self.__ef_construction: int | None = ef_construction
        self.__ef_search: int | None = ef_search
        self.__maintenance_work_mem: str | None = maintenance_work_mem
        self.__max_parallel_maintenance_workers: int = max_parallel_maintenance_workers

    def index_param(self) -> dict | None:
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

    def search_param(self) -> dict | None:
        param = {
            "metric_operator": _metric_operator[self.__metric_type],
            "set": {}
        }
        if self.__ef_search is not None:
            param["set"]["hnsw.ef_search"] = self.__ef_search
        return param


class PgvectorIVFFlatConfig(BaseIndexConfig):
    def __init__(self, metric_type: MetricType, lists: int, probes: int | None = None,
                 max_parallel_maintenance_workers: int | None = None):
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type = metric_type
        self.__lists: int = lists
        self.__probes: int | None = probes
        self.__max_parallel_maintenance_workers: int = max_parallel_maintenance_workers

    def index_param(self) -> dict | None:
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

    def search_param(self) -> dict | None:
        param = {
            "metric_operator": _metric_operator[self.__metric_type],
            "set": {}
        }
        if self.__probes is not None:
            param["set"]["ivfflat.probes"] = self.__probes
        return param


_distance_metric = {
    MetricType.L2: "vector_l2_ops",
    MetricType.IP: "vector_ip_ops",
    MetricType.COSINE: "vector_cosine_ops"
}

_metric_operator = {
    MetricType.L2: "<->",
    MetricType.IP: "<#>",
    MetricType.COSINE: "<=>"
}
