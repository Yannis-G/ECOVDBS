from .runner_config import HNSWTask, IndexTime, InsertConfig, QueryConfig
from ..utility import time_it
from ...client.base.base_client import BaseClient
from ...dataset.dataset import Dataset


class HNSWRunner:
    def __init__(self, hnsw_task: HNSWTask):
        self.__insert_runner = InsertRunner(hnsw_task.client, hnsw_task.insert_config, hnsw_task.dataset)
        self.__query_runner = QueryRunner(hnsw_task.client, hnsw_task.query_config, hnsw_task.dataset)

    def run(self):
        self.__insert_runner.run()
        self.__query_runner.run()


class InsertRunner:
    def __init__(self, client: BaseClient, config: InsertConfig, dataset: Dataset):
        self.__client = client
        self.__index_time = config.index_time
        self.__dataset = dataset

    def run(self):
        if self.__index_time == IndexTime.PRE_INDEX:
            _, t_index = self.create_index()
            _, t_insert = self.insert(self.__dataset.data_vectors)
        elif self.__index_time == IndexTime.POST_INDEX:
            _, t_insert = self.insert(self.__dataset.data_vectors)
            _, t_index = self.create_index()
        elif self.__index_time == IndexTime.NO_INDEX:
            _, t_insert = self.insert(self.__dataset.data_vectors)
            t_index = 0
        else:
            raise ValueError("Invalid index time")
        return t_index + t_insert

    @time_it
    def insert(self, embeddings: list[list[float]]):
        self.__client.insert(embeddings)

    @time_it
    def create_index(self):
        self.__client.create_index()


class QueryRunner:
    def __init__(self, client: BaseClient, config: QueryConfig, dataset: Dataset):
        self.__client: BaseClient = client
        self.__config = config
        self.__dataset: Dataset = dataset
        self.k: int = len(self.__dataset.ground_truth_neighbors[0])
        self.total_time: float = 0
        self.total_recall: float = 0
        self.num_queries: int = len(self.__dataset.query_vectors)

    def run(self):
        for ef in self.__config.ef_search:
            self.__config.index_config.change_ef_search(ef)
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
