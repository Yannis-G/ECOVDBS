import logging
from typing import Optional, Callable

from .result_config import (InsertRunnerResult, HNSWQueryEFResult, HNSWQueryModeResult, HNSWQueryRunnerResult,
                            HNSWRunnerResult)
from .task_config import HNSWTask, IndexTime, InsertConfig, HNSWQueryConfig, QueryMode
from .utility import time_it
from ..client.base_client import BaseClient
from ..client.base_config import BaseHNSWConfig
from ..dataset.dataset import Dataset

log = logging.getLogger(__name__)


class HNSWRunner:
    """
    Runner class for HNSW tasks, coordinating insertion and query operations.
    """

    def __init__(self, hnsw_task: HNSWTask):
        """
        Initialize the HNSWRunner with a given HNSW task.

        :param hnsw_task: The HNSW task configuration (see :class:`HNSWTask`).
        """
        self.__client = hnsw_task.client
        self.__index_config = hnsw_task.query_config.index_config
        self.__insert_runner = InsertRunner(hnsw_task.client, hnsw_task.insert_config, hnsw_task.dataset)
        self.__query_runner = HNSWQueryRunner(hnsw_task.client, hnsw_task.query_config, hnsw_task.dataset)

    def run(self):
        """
        Run the HNSW task, performing both insert and query operations.

        :return: Results of the HNSW task (see :class:`HNSWRunnerResult`).
        """
        insert_result = self.__insert_runner.run()
        query_result = self.__query_runner.run()
        index_size = self.__client.index_storage()
        disk_size = self.__client.disk_storage()
        return HNSWRunnerResult(self.__client, self.__index_config, insert_result, query_result, index_size, disk_size)


class InsertRunner:
    """
    Runner class for handling insert operations in the HNSW task.
    """

    def __init__(self, client: BaseClient, config: InsertConfig, dataset: Dataset):
        """
        Initialize the InsertRunner with the client, configuration, and dataset.

        :param client: The client to interact with the database (see :class:`BaseClient`).
        :param config: Configuration for the insert operation (see :class:`InsertConfig`).
        :param dataset: The dataset to be used for the insert operation (see :class:`Dataset`).
        """
        self.__client: BaseClient = client
        self.__index_time: IndexTime = config.index_time
        self.__data_vectors: list[list[float]] = dataset.data_vectors
        self.__metadata: Optional[
            list[str]] = dataset.metadata if QueryMode.FILTERED_QUERY in config.query_modes else None

    def run(self):
        """
        Run the insert operation according to the configuration.

        :return: Result of the insert operation (see :class:`InsertRunnerResult`).
        """
        log.info("Start InsertRunner for client %s", type(self.__client).__name__)
        if self.__index_time == IndexTime.PRE_INDEX:
            _, t_index = self.__create_index()
            _, t_insert = self.__insert(self.__data_vectors, self.__metadata)
        elif self.__index_time == IndexTime.POST_INDEX:
            _, t_insert = self.__insert(self.__data_vectors, self.__metadata)
            _, t_index = self.__create_index()
        elif self.__index_time == IndexTime.NO_INDEX:
            _, t_insert = self.__insert(self.__data_vectors, self.__metadata)
            t_index = 0
        else:
            raise ValueError("Invalid index time")
        return InsertRunnerResult(t_insert + t_index)

    @time_it
    def __insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None) -> None:
        """
        Insert embeddings into the database.

        :param embeddings: List of data vectors to be inserted.
        :param metadata: Optional list of metadata corresponding to the data vectors.
        """
        log.info("Insert %d embeddings", len(embeddings))
        self.__client.insert(embeddings, metadata)

    @time_it
    def __create_index(self) -> None:
        """
        Create the HNSW index in the database.
        """
        log.info("Create index")
        self.__client.create_index()


class HNSWQueryRunner:
    """
    Runner class for handling query operations in the HNSW task.
    """

    def __init__(self, client: BaseClient, config: HNSWQueryConfig, dataset: Dataset) -> None:
        """
        Initialize the HNSWQueryRunner with the client, configuration, and dataset.

        :param client: The client to interact with the database (see :class:`BaseClient`).
        :param config: Configuration for the query operation (see :class:`HNSWQueryConfig`).
        :param dataset: The dataset to be used for the query operation (see :class:`Dataset`).
        """
        self.__client: BaseClient = client
        self.__ef_search: list[int] = config.ef_search
        self.__index_config: BaseHNSWConfig = config.index_config
        self.__query_mode: list[QueryMode] = config.query_modes
        self.__query_vectors: list[list[float]] = dataset.query_vectors
        self.__ground_truth_neighbors: list[list[int]] = dataset.ground_truth_neighbors
        self.__keyword_filters: Optional[list[str]] = dataset.keyword_filter
        self.__distances: Optional[list[float]] = dataset.distance
        self.__k: int = len(self.__ground_truth_neighbors[0])
        self.__total_time: float = 0
        self.__total_recall: float = 0
        self.__num_queries: int = len(self.__query_vectors)

    def run(self) -> HNSWQueryRunnerResult:
        """
        Run the query operation according to the configuration.

        :return: Results of the query operation (see :class:`HNSWQueryRunnerResult`).
        """
        log.info("Start QueryRunner for client %s", type(self.__client).__name__)
        mode_results: list[HNSWQueryModeResult] = []
        for query_mode in self.__query_mode:
            mode_results.append(self.__run_mode(query_mode))
        return HNSWQueryRunnerResult(mode_results)

    def __run_mode(self, query_mode: QueryMode) -> HNSWQueryModeResult:
        """
        Run queries for a specific query mode.

        :param query_mode: The query mode to run (see :class:`QueryMode`).
        :return: Results of the queries for the specific mode (see :class:`HNSWQueryModeResult`).
        """
        log.info(f"Run %d queries for mode %s", self.__num_queries * len(self.__ef_search), query_mode.name)
        extended_list, query_func = self.__get_mode_params(query_mode)
        ef_results: list[HNSWQueryEFResult] = []
        for ef in self.__ef_search:
            self.__index_config.change_ef_search(ef)
            self.total_time = 0
            self.total_recall = 0

            log.info("Run %d queries for ef %d", self.__num_queries, ef)
            if query_mode == QueryMode.QUERY:
                _, total_duration = self.__run_queries()
            else:
                _, total_duration = self.__run_queries_extended(query_func, extended_list)

            avg_recall: float = self.total_recall / self.__num_queries
            avg_query_time: float = self.total_time / self.__num_queries
            queries_per_second: float = self.__num_queries / total_duration

            ef_results.append(HNSWQueryEFResult(ef, avg_recall, avg_query_time, queries_per_second, total_duration,
                                                self.__num_queries, self.__k))
        return HNSWQueryModeResult(query_mode, ef_results)

    def __get_mode_params(self, query_mode):
        """
        Get the query parameters based on the query mode.

        :param query_mode: The query mode to run (see :class:`QueryMode`).
        :return: A tuple containing the extended list and query function.
        """
        if query_mode == QueryMode.QUERY:
            query_func = self.__query
            extended_list = None
        elif query_mode == QueryMode.FILTERED_QUERY and self.__keyword_filters is not None:
            query_func = self.__filtered_query
            extended_list = self.__keyword_filters
        elif query_mode == QueryMode.RANGED_QUERY and self.__distances is not None:
            query_func = self.__ranged_query
            extended_list = self.__distances
        else:
            raise ValueError("Invalid query mode")
        return extended_list, query_func

    @time_it
    def __run_queries(self) -> None:
        """
        Run the standard queries.
        """
        for q, gt in zip(self.__query_vectors, self.__ground_truth_neighbors):
            res, t = self.__query(q, self.__k)
            recall = len(set(gt) & set(res)) / self.__k
            self.total_recall += recall
            self.total_time += t

    @time_it
    def __run_queries_extended(self, query_func: Callable[[list[float], int, str | float], list[int]],
                               extended: list[str | float]) -> None:
        """
        Run extended queries (filtered or ranged).

        :param query_func: The query function to use.
        :param extended: List of extended parameters (e.g., keywords or distances).
        """
        for q, gt, e in zip(self.__query_vectors, self.__ground_truth_neighbors, extended):
            res, t = query_func(q, self.__k, e)
            recall = len(set(gt) & set(res)) / self.__k
            self.total_recall += recall
            self.total_time += t

    @time_it
    def __query(self, query: list[float], k: int) -> list[int]:
        """
        Perform a standard query.

        :param query: The query vector.
        :param k: The number of nearest neighbors to retrieve.
        :return: List of retrieved nearest neighbors.
        """
        return self.__client.query(query, k)

    @time_it
    def __filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        """
        Perform a filtered query.

        :param query: The query vector.
        :param k: The number of nearest neighbors to retrieve.
        :param keyword_filter: The keyword filter to apply.
        :return: List of retrieved nearest neighbors.
        """
        return self.__client.filtered_query(query, k, keyword_filter)

    @time_it
    def __ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        """
        Perform a ranged query.

        :param query: The query vector.
        :param k: The number of nearest neighbors to retrieve.
        :param distance: The maximum distance for the query.
        :return: List of retrieved nearest neighbors.
        """
        return self.__client.ranged_query(query, k, distance)
