from ..client.base.base_config import MetricType
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    dimension: int
    metric_type: MetricType
    data_vectors: list[list[float]]
    query_vectors: list[list[float]]
    ground_truth_neighbors: list[list[int]]


@dataclass(frozen=True)
class FilteredDataset(Dataset):
    metadata: list[str]
    keyword_filter: list[str]


@dataclass(frozen=True)
class RangedDataset(Dataset):
    distance: list[float]
