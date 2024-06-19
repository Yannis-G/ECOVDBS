from ..client.base.base_config import MetricType
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """
    Represents a dataset used for vector search tasks.

    Attributes:
        dimension: The dimensionality of the vectors.
        metric_type: The type of metric used for distance calculation.
        data_vectors: The list of data vectors.
        query_vectors: The list of query vectors.
        ground_truth_neighbors: The list of ground truth neighbors for each query vector.
    """
    dimension: int
    metric_type: MetricType
    data_vectors: list[list[float]]
    query_vectors: list[list[float]]
    ground_truth_neighbors: list[list[int]]


@dataclass(frozen=True)
class FilteredDataset(Dataset):
    """
    Represents a dataset with additional metadata and keyword filtering.

    Attributes:
        metadata (list[str]): List of metadata associated with each data vector.
        keyword_filter (list[str]): List of keywords used for filtering the data vectors.
    """
    metadata: list[str]
    keyword_filter: list[str]


@dataclass(frozen=True)
class RangedDataset(Dataset):
    """
    Represents a dataset with additional distance attributes.

    Attributes:
        distance (list[float]): List of distances used in the ranged search.
    """
    distance: list[float]
