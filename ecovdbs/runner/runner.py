from .utility import time_it
from ..client.base.base_client import BaseClient
from ..dataset.dataset import Dataset, FilteredDataset, RangedDataset


class Runner:
    def __init__(self, config):
        self.__config = config


class InsertRunner:
    def __init__(self, client: BaseClient, config, dataset: Dataset):
        self.__client = client
        self.__config = config
        self.__dataset = dataset

    def run(self):
        self.insert(self.__dataset.data_vectors)

    @time_it
    def insert(self, embeddings: list[list[float]]):
        self.__client.insert(embeddings)


class FilteredInsertRunner:
    def __init__(self, client: BaseClient, config, dataset: FilteredDataset):
        self.__client = client
        self.__config = config
        self.__dataset = dataset

    def run(self):
        _, t = self.insert(self.__dataset.data_vectors, self.__dataset.metadata)
        print(t)

    @time_it
    def insert(self, embeddings: list[list[float]], metadata: list[str]):
        self.__client.insert(embeddings, metadata)


class IndexRunner:
    def __init__(self, client: BaseClient, config, dataset: Dataset):
        self.__client = client
        self.__config = config
        self.__dataset = dataset

    def run(self):
        pass

    @time_it
    def create_index(self):
        self.__client.create_index()


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
