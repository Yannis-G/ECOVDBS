import time

from .runner_config import HNSWTask
from ...client.base.base_client import BaseClient
from ...client.base.base_config import BaseHNSWConfig
from ...dataset.dataset import Dataset
from ..utility import time_it


class HNSWRunner:
    def __init__(self, hnsw_task: HNSWTask):
        self.__client = hnsw_task.client
        self.__index_config = hnsw_task.index_config
        self.__dataset = hnsw_task.dataset
        self.__insert_runner = InsertRunner(self.__client, None, self.__dataset)
        self.__query_runner = QueryRunner(self.__client, (hnsw_task.ef_search, self.__index_config), self.__dataset)

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
        return t_index + t_insert

    @time_it
    def insert(self, embeddings: list[list[float]]):
        self.__client.insert(embeddings)

    @time_it
    def create_index(self):
        self.__client.create_index()


class QueryRunner:
    def __init__(self, client: BaseClient, config: tuple[list, BaseHNSWConfig], dataset: Dataset):
        self.__client: BaseClient = client
        self.__config = config
        self.__dataset: Dataset = dataset
        self.k: int = len(self.__dataset.ground_truth_neighbors[0])
        self.total_time: float = 0
        self.total_recall: float = 0
        self.num_queries: int = len(self.__dataset.query_vectors)

    def run(self):
        for ef in self.__config[0]:
            self.__config[1].change_ef_search(ef)
            self.total_time = 0
            self.total_recall = 0

            _, total_duration = self.one()

            avg_recall = self.total_recall / self.num_queries
            avg_query_time = self.total_time / self.num_queries
            queries_per_second = self.num_queries / total_duration

            print(f"ef: {ef}")
            print(f'Average Recall: {avg_recall}')
            print(f'Average Query Time: {avg_query_time}')
            print(f'Queries per Second: {queries_per_second}')

    @time_it
    def one(self):
        for q, gt in zip(self.__dataset.query_vectors, self.__dataset.ground_truth_neighbors):
            res, t = self.query(q, self.k)
            recall = len(set(gt) & set(res)) / len(res)
            self.total_recall += recall
            self.total_time += t

    @time_it
    def query(self, query: list[float], k: int) -> list[int]:
        return self.__client.query(query, k)

    @time_it
    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        return self.__client.filtered_query(query, k, keyword_filter)

    @time_it
    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        return self.__client.ranged_query(query, k, distance)
