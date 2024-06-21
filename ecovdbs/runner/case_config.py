import os
from dataclasses import dataclass, field
from enum import Enum

from ..client.base.base_config import MetricType
from ..dataset.dataset import Dataset
from ..dataset.utility import fvecs_read, ivecs_read


class IndexTime(Enum):
    """
    Enum class for the time at which the index is created.

    Attributes:
        PRE_INDEX: Indicates the index is created before insertion.
        POST_INDEX: Indicates the index is created after insertion.
        NO_INDEX: Indicates no index is created.
    """
    PRE_INDEX = 0
    POST_INDEX = 1
    NO_INDEX = 2


class QueryMode(Enum):
    """
    Enum class for the query mode.

    Attributes:
        QUERY: Regular query mode.
        FILTERED_QUERY: Query mode with filtering applied.
        RANGED_QUERY: Query mode with a specified range.
    """
    QUERY = 0
    FILTERED_QUERY = 1
    RANGED_QUERY = 2


@dataclass
class HNSWConfig:
    """
    Configuration class for HNSW (Hierarchical Navigable Small World) algorithm parameters.

    Attributes:
        M: The number of bidirectional links created for every new element during construction. Default is 24.
        ef_construction: The size of the dynamic list for the nearest neighbors (used during the index construction).
            Default is 200.
        ef_search: A list of sizes for the dynamic list for the nearest neighbors (used during search). Default is a
            list of values [10, 20, 40, 80, 120, 200, 400, 800].
    """
    M: int = 24
    ef_construction: int = 200
    ef_search: list[int] = field(default_factory=lambda: [10, 20, 40, 80, 120, 200, 400, 800])


@dataclass(init=False)
class HNSWCase:
    """
    Case class for running HNSW tasks, including the dataset and configuration.

    Attributes:
        dataset: The dataset to be used for the case (see :class:`Dataset`).
        hnsw_config: Configuration for the HNSW algorithm (see :class:`HNSWConfig`).
        index_time: The time at which the index is created (see :class:`IndexTime`).
        query_modes: A list of query modes to be used (see :class:`QueryMode`).
    """
    dataset: Dataset
    hnsw_config: HNSWConfig
    index_time: IndexTime
    query_modes: list[QueryMode]


@dataclass(init=False)
class TestCase(HNSWCase):
    """
    Test case class for running HNSW tasks with a specific dataset and configuration.
    """
    dataset = Dataset(128, MetricType.L2, fvecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_base.fvecs")), fvecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_query.fvecs")), ivecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_groundtruth.ivecs")),
                      [str(i) for i in range(10_000)], [str(i) for i in range(10_000)])
    hnsw_config = HNSWConfig()
    index_time = IndexTime.PRE_INDEX
    query_modes = [QueryMode.FILTERED_QUERY, QueryMode.QUERY]
