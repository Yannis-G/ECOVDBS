import os
from dataclasses import dataclass, field
from enum import Enum

from ...client.base.base_client import BaseClient
from ...client.base.base_config import BaseHNSWConfig, MetricType
from ...dataset.dataset import Dataset
from ...dataset.utility import fvecs_read, ivecs_read


class IndexTime(Enum):
    """
    Enum class for the time at which the index is created.
    """
    PRE_INDEX = 0
    POST_INDEX = 1
    NO_INDEX = 2


class QueryMode(Enum):
    """
    Enum class for the query mode.
    """
    QUERY = 0
    FILTERED_QUERY = 1
    RANGED_QUERY = 2


@dataclass(frozen=True)
class InsertConfig:
    """
    Configuration class for the insertion operation.

    Attributes:
        index_time: The time at which the index is created (see :class:`IndexTime`).
        query_modes: The query modes (see :class:`QueryMode`).
    """
    index_time: IndexTime
    query_modes: list[QueryMode]


@dataclass(frozen=True)
class QueryConfig:
    """
    Configuration class for the query operation.

    Attributes:
        ef_search: A list of sizes for the dynamic list for the nearest neighbors (used during search).
        index_config: Configuration for the HNSW index (see :class:`BaseHNSWConfig`).
        query_modes: The query modes (see :class:`QueryMode`).
    """
    ef_search: list[int]
    index_config: BaseHNSWConfig
    query_modes: list[QueryMode]


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
class HNSWTask:
    """
    Task class for HNSW operations, including configuration, client, and dataset.

    Attributes:
        client: The client to interact with the HNSW index (see :class:`BaseClient`).
        dataset: The dataset to be used for the task (see :class:`Dataset`).
        insert_config: Configuration for the insertion operation (see :class:`InsertConfig`).
        query_config: Configuration for the query operation (see :class:`QueryConfig`).
    """
    client: BaseClient
    dataset: Dataset
    insert_config: InsertConfig
    query_config: QueryConfig


@dataclass(init=False)
class HNSWCase:
    """
    Case class for running HNSW tasks, including the dataset and configuration.

    Attributes:
        dataset: The dataset to be used for the case (see :class:`Dataset`).
        hnsw_config: Configuration for the HNSW algorithm (see :class:`HNSWConfig`).
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
