from .runner_config import HNSWTask
from ...client.base.base_client import BaseClient
from ...dataset.dataset import Dataset, FilteredDataset, RangedDataset
from ..utility import time_it


class HNSWRunner:
    def __init__(self, hnsw_task: HNSWTask):
        self.__client = hnsw_task.client
        self.__index_config = hnsw_task.index_config
        self.__dataset = hnsw_task.dataset
        self.__insert_runner = InsertRunner(self.__client, None, self.__dataset)
        self.__query_runner = QueryRunner(self.__client, None, self.__dataset)

    def run(self):
        self.__insert_runner.run()
        self.__query_runner.run()


class InsertRunner:
    def __init__(self, client: BaseClient, config, dataset: Dataset):
        self.__client = client
        self.__config = config
        self.__dataset = dataset

    def run(self):
        _, t_index = self.create_index()
        _, t_insert = self.insert(self.__dataset.data_vectors)
        print("Index+Insert Time", t_index, t_insert)

    @time_it
    def insert(self, embeddings: list[list[float]]):
        self.__client.insert(embeddings)

    @time_it
    def create_index(self):
        self.__client.create_index()


class FilteredInsertRunner:
    def __init__(self, client: BaseClient, config, dataset: FilteredDataset):
        self.__client = client
        self.__config = config
        self.__dataset = dataset

    def run(self):
        _, t_index = self.create_index()
        _, t_insert = self.insert(self.__dataset.data_vectors, self.__dataset.metadata)
        print("Index+Insert Time", t_index, t_insert)

    @time_it
    def insert(self, embeddings: list[list[float]], metadata: list[str]):
        self.__client.insert(embeddings, metadata)


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
