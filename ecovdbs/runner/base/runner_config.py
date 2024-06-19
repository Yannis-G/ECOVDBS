import os

from ...dataset.dataset import Dataset, FilteredDataset
from ...client.base.base_config import BaseHNSWConfig, MetricType
from ...client.base.base_client import BaseClient
from dataclasses import dataclass, field
from ...dataset.utility import fvecs_read, ivecs_read


@dataclass
class HNSWConfig:
    M: int = 24
    ef_construction: int = 200
    ef_search: list[int] = field(default_factory=lambda: [10, 20, 40, 80, 120, 200, 400, 800])


@dataclass(init=False)
class HNSWTask:
    index_config: BaseHNSWConfig
    client: BaseClient
    dataset: Dataset


@dataclass(init=False)
class HNSWCase:
    dataset: Dataset
    hnsw_config: HNSWConfig


@dataclass(init=False)
class TestCase(HNSWCase):
    dataset = FilteredDataset(128, MetricType.L2, fvecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_base.fvecs")), fvecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_query.fvecs")), ivecs_read(
        os.path.join(os.getenv("PYTHONPATH").split(";")[0], "data/siftsmall/siftsmall_groundtruth.ivecs")),
                              [str(i) for i in range(10_000)], [str(i) for i in range(10_000)])
    hnsw_config = HNSWConfig()
