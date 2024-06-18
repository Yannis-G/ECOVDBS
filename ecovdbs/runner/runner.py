from .utility import time_it
from ..client.base.base_client import BaseClient
from ..dataset.dataset import Dataset, FilteredDataset, RangedDataset


class QueryRunner:
    def __init__(self, client: BaseClient, config, dataset: Dataset):
        self.__client: BaseClient = client
        self.__config = config
        self.__dataset: Dataset = dataset

    def run(self):
        k = len(self.__dataset.ground_truth_neighbors[0])
        for q, gt in zip(self.__dataset.query_vectors, self.__dataset.ground_truth_neighbors):
            res, t = self.query(q, k)
            recall = len(set(gt) & set(res)) / len(res)
            print(recall, t)
        # Run all query
        # Metric
        # Increase ef_search
        # Repeat

    @time_it
    def query(self, query: list[float], k: int) -> list[int]:
        return self.__client.query(query, k)


class FilteredQueryRunner(QueryRunner):
    def __init__(self, client: BaseClient, config, dataset: FilteredDataset):
        super().__init__(client, config, dataset)

    @time_it
    def query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        return self.__client.filtered_query(query, k, keyword_filter)


class RangedQueryRunner(QueryRunner):
    def __init__(self, client: BaseClient, config, dataset: RangedDataset):
        super().__init__(client, config, dataset)

    @time_it
    def query(self, query: list[float], k: int, distance: float) -> list[int]:
        return self.__client.ranged_query(query, k, distance)
