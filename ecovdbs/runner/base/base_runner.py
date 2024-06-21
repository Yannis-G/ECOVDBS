from typing import Optional

from .runner_config import HNSWTask, IndexTime, InsertConfig, QueryConfig, QueryMode
from ..utility import time_it
from ...client.base.base_client import BaseClient
from ...client.base.base_config import BaseHNSWConfig
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
        self.__client: BaseClient = client
        self.__index_time: IndexTime = config.index_time
        self.__data_vectors: list[list[float]] = dataset.data_vectors
        self.__metadata: Optional[
            list[str]] = dataset.metadata if config.query_mode == QueryMode.FILTERED_QUERY else None

    def run(self):
        if self.__index_time == IndexTime.PRE_INDEX:
            _, t_index = self.create_index()
            _, t_insert = self.insert(self.__data_vectors, self.__metadata)
        elif self.__index_time == IndexTime.POST_INDEX:
            _, t_insert = self.insert(self.__data_vectors, self.__metadata)
            _, t_index = self.create_index()
        elif self.__index_time == IndexTime.NO_INDEX:
            _, t_insert = self.insert(self.__data_vectors, self.__metadata)
            t_index = 0
        else:
            raise ValueError("Invalid index time")
        return t_index + t_insert

    @time_it
    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None) -> None:
        self.__client.insert(embeddings, metadata)

    @time_it
    def create_index(self) -> None:
        self.__client.create_index()


class QueryRunner:
    def __init__(self, client: BaseClient, config: QueryConfig, dataset: Dataset):
        self.__client: BaseClient = client
        self.__ef_search: list[int] = config.ef_search
        self.__index_config: BaseHNSWConfig = config.index_config
        self.__query_mode: QueryMode = config.query_mode
        self.__query_vectors: list[list[float]] = dataset.query_vectors
        self.__ground_truth_neighbors: list[list[int]] = dataset.ground_truth_neighbors
        self.__keyword_filters: Optional[list[str]] = dataset.keyword_filter
        self.__distances: Optional[list[float]] = dataset.distance
        self.k: int = len(self.__ground_truth_neighbors[0])
        self.total_time: float = 0
        self.total_recall: float = 0
        self.num_queries: int = len(self.__query_vectors)

    def run(self):
        if self.__query_mode == QueryMode.QUERY:
            query_func = self.query
        elif self.__query_mode == QueryMode.FILTERED_QUERY and self.__keyword_filters is not None:
            query_func = self.filtered_query
            third = self.__keyword_filters
        elif self.__query_mode == QueryMode.RANGED_QUERY and self.__distances is not None:
            query_func = self.ranged_query
            third = self.__distances
        else:
            raise ValueError("Invalid query mode")
        for ef in self.__ef_search:
            self.__index_config.change_ef_search(ef)
            self.total_time = 0
            self.total_recall = 0

            if self.__query_mode == QueryMode.QUERY:
                _, total_duration = self.one()
            else:
                _, total_duration = self.two(query_func, third)

            avg_recall = self.total_recall / self.num_queries
            avg_query_time = self.total_time / self.num_queries
            queries_per_second = self.num_queries / total_duration

            print(f"ef: {ef}")
            print(f'Average Recall: {avg_recall}')
            print(f'Average Query Time: {avg_query_time}')
            print(f'Queries per Second: {queries_per_second}')

    @time_it
    def one(self):
        for q, gt in zip(self.__query_vectors, self.__ground_truth_neighbors):
            res, t = self.query(q, self.k)
            recall = len(set(gt) & set(res)) / len(res)
            self.total_recall += recall
            self.total_time += t

    @time_it
    def two(self, query_func, third):
        for q, gt, t in zip(self.__query_vectors, self.__ground_truth_neighbors, third):
            res, t = query_func(q, self.k, t)
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
