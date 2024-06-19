from typing import Optional

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
        metadata: Optional list of metadata for the data vectors.
        keyword_filter: Optional list of keywords for filtering the data vectors.
        distance: Optional list of distances corresponding to the query vectors.
    """
    dimension: int
    metric_type: MetricType
    data_vectors: list[list[float]]
    query_vectors: list[list[float]]
    ground_truth_neighbors: list[list[int]]
    metadata: Optional[list[str]] = None
    keyword_filter: Optional[list[str]] = None
    distance: Optional[list[float]] = None
