from dataclasses import dataclass

from ..base_config import BaseConfig, BaseIndexConfig, IndexType, MetricType, BaseHNSWConfig


@dataclass(frozen=True)
class MilvusConfig(BaseConfig):
    """
    Configuration class for Milvus database clients.

    Attributes:
        connection_uri: The connection URI for the Milvus server. Defaults to "http://localhost:19530".
    """
    connection_uri: str = "http://localhost:19530"


class MilvusAutoIndexConfig(BaseIndexConfig):
    """
    Configuration class for AutoIndex type index in Milvus.
    """

    def __init__(self, metric_type: MetricType):
        """
        Initialize MilvusAutoIndexConfig.

        :param metric_type: The metric type for the index.
        """
        self.__index_type: IndexType = IndexType.AUTOINDEX
        self.__metric_type: MetricType = metric_type

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {},
        }


class MilvusFlatConfig(BaseIndexConfig):
    """
    Configuration class for Flat type index in Milvus.
    """

    def __init__(self, metric_type: MetricType):
        """
        Initialize MilvusFlatConfig.

        :param metric_type: The metric type for the index.
        """
        self.__index_type: IndexType = IndexType.Flat
        self.__metric_type: MetricType = metric_type

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {},
        }


class MilvusIVFFlatIndex(BaseIndexConfig):
    """
    Configuration class for IVF_FLAT type index in Milvus.
    """

    def __init__(self, metric_type: MetricType, nlist: int = 128, nprobe: int = 8):
        """
        Initialize MilvusIVFFlatIndex.

        :param nlist: Number of cluster units.
        :param nprobe: Number of units to query.
        """
        self.__index_type: IndexType = IndexType.IVFFlat
        self.__metric_type: MetricType = metric_type

        assert 0 < nlist <= 65536
        self.__nlist: int = nlist
        assert 0 < nprobe <= self.__nlist
        self.__nprobe: int = nprobe

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {"nlist": self.__nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {"nprobe": self.__nprobe},
        }


class MilvusIVFSQ8Index(BaseIndexConfig):
    """
    Configuration class for IVF_SQ8 type index in Milvus.
    """

    def __init__(self, metric_type: MetricType, nlist: int, nprobe: int = 8):
        """
        Initialize MilvusIVFSQ8Index.

        :param nlist: Number of cluster units.
        :param nprobe: Number of units to query.
        """
        self.__index_type: IndexType = IndexType.IVFSQ8
        self.__metric_type: MetricType = metric_type

        assert 0 < nlist <= 65536
        self.__nlist: int = nlist
        assert 0 < nprobe <= self.__nlist
        self.__nprobe: int = nprobe

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {"nlist": self.__nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {"nprobe": self.__nprobe},
        }


class MilvusIVFPQIndex(BaseIndexConfig):
    """
    Configuration class for IVF_PQ type index in Milvus.
    """

    def __init__(self, metric_type: MetricType, nlist: int, m: int, nbits: int = 8, nprobe: int = 8):
        """
        Initialize MilvusIVFPQIndex.

        :param nlist: Number of cluster units.
        :param m: Number of factors of product quantization.
        :param nbits: Number of bits in which each low-dimensional vector is stored.
        :param nprobe: Number of units to query.
        """
        self.__index_type: IndexType = IndexType.IVFPQ
        self.__metric_type: MetricType = metric_type

        assert 0 < nlist <= 65536
        self.__nlist: int = nlist
        self.__m: int = m
        assert 0 < nbits <= 16
        self.__nbits: int = nbits
        assert 0 < nprobe <= self.__nlist
        self.__nprobe: int = nprobe

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {"nlist": self.__nlist, "m": self.__m, "nbits": self.__nbits},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {"nprobe": self.__nprobe},
        }


class MilvusSCANNIndex(BaseIndexConfig):
    """
    Configuration class for SCANN type index in Milvus.
    """

    def __init__(self, metric_type: MetricType, nlist: int, nprobe: int, reorder_k: int, with_raw_data: bool = True):
        """
        Initialize MilvusSCANNIndex.

        :param nlist: Number of cluster units.
        :param nprobe: Number of units to query.
        :param reorder_k: Number of candidate units to query.
        :param with_raw_data: Whether to include the raw data in the index.
        """
        self.__index_type: IndexType = IndexType.IVFPQ
        self.__metric_type: MetricType = metric_type

        assert 0 < nlist <= 65536
        self.__nlist: int = nlist
        self.__with_raw_data: bool = with_raw_data
        assert 0 < nprobe <= self.__nlist
        self.__nprobe: int = nprobe
        self.__reorder_k: int = reorder_k

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {"nlist": self.__nlist, "with_raw_data": self.__with_raw_data},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {"nprobe": self.__nprobe, "reorder_k": self.__reorder_k},
        }


class MilvusHNSWConfig(BaseHNSWConfig):
    """
    Configuration class for HNSW type index in Milvus.
    """

    def __init__(self, metric_type: MetricType, M: int, efConstruction: int, ef: int):
        """
        Initialize MilvusHNSWConfig.

        :param M: M defines tha maximum number of outgoing connections in the graph. Higher M leads to higher
            accuracy/run_time at fixed ef/efConstruction.
        :param efConstruction: ef_construction controls index search speed/build speed tradeoff. Increasing the
            efConstruction parameter may enhance index quality, but it also tends to lengthen the indexing time.
        :param ef: Parameter controlling query time/accuracy trade-off. Higher ef leads to more accurate but slower
            search.
        """
        self.__index_type: IndexType = IndexType.HNSW
        self.__metric_type: MetricType = metric_type

        assert 2 <= M <= 2048
        self.__M: int = M
        assert efConstruction > 0
        self.__efConstruction: int = efConstruction
        self.__ef: int = ef

    def index_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "index_type": self.__index_type.value,
            "params": {"M": self.__M, "efConstruction": self.__efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.__metric_type.value,
            "params": {"ef": self.__ef},
        }

    def change_ef_search(self, ef: int) -> None:
        self.__ef = ef
