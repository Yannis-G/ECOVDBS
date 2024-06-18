from ..client.base.base_config import MetricType


class Dataset:
    def __init__(self, dimension: int, metric_type: MetricType, data_vectors: list[list[float]],
                 query_vectors: list[list[float]], ground_truth_neighbors: list[list[int]]):
        self.__dimension: int = dimension
        self.__metric_type: MetricType = metric_type
        self.__data_vectors: list[list[float]] = data_vectors
        self.__query_vectors: list[list[float]] = query_vectors
        self.__ground_truth_neighbors: list[list[int]] = ground_truth_neighbors

    @property
    def dimension(self) -> int:
        return self.__dimension

    @property
    def metric_type(self) -> MetricType:
        return self.__metric_type

    @property
    def data_vectors(self) -> list[list[float]]:
        return self.__data_vectors

    @property
    def query_vectors(self) -> list[list[float]]:
        return self.__query_vectors

    @property
    def ground_truth_neighbors(self) -> list[list[int]]:
        return self.__ground_truth_neighbors


class FilteredDataset(Dataset):
    def __init__(self, dimension: int, metric_type: MetricType, data_vectors: list[list[float]],
                 query_vectors: list[list[float]], ground_truth_neighbors: list[list[int]], metadata: list[str],
                 keyword_filter: list[str]):
        super().__init__(dimension, metric_type, data_vectors, query_vectors, ground_truth_neighbors)
        self.__metadata: list[str] = metadata
        self.__keyword_filter: list[str] = keyword_filter

    @property
    def metadata(self) -> list[str]:
        return self.__metadata

    @property
    def keyword_filter(self) -> list[str]:
        return self.__keyword_filter


class RangedDataset(Dataset):
    def __init__(self, dimension: int, metric_type: MetricType, data_vectors: list[list[float]],
                 query_vectors: list[list[float]], ground_truth_neighbors: list[list[int]], distance: list[float]):
        super().__init__(dimension, metric_type, data_vectors, query_vectors, ground_truth_neighbors)
        self.__distance: list[float] = distance

    @property
    def distance(self) -> list[float]:
        return self.__distance
