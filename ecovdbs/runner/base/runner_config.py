import os

from ...dataset.dataset import Dataset
from ...client.base.base_config import BaseHNSWConfig, MetricType
from ...client.base.base_client import BaseClient
from dataclasses import dataclass, field
from ...dataset.utility import fvecs_read, ivecs_read


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
        index_config: Configuration for the HNSW index (see :class:`BaseHNSWConfig`).
        client: The client to interact with the HNSW index (see :class:`BaseClient`).
        dataset: The dataset to be used for the task (see :class:`Dataset`).
        ef_search: A list of sizes for the dynamic list for the nearest neighbors (used during search).
    """
    index_config: BaseHNSWConfig
    client: BaseClient
    dataset: Dataset
    ef_search: list[int]


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
