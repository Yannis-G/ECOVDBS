from dataclasses import dataclass

from .task_config import QueryMode
from ...client.base.base_client import BaseClient
from ...client.base.base_config import BaseHNSWConfig


@dataclass(frozen=True)
class InsertRunnerResult:
    """
    Data class representing the result of an insertion operation.

    Attributes:
        t_insert_index: The time taken to insert the data and create the index.
    """
    t_insert_index: float


@dataclass(frozen=True)
class HNSWQueryEFResult:
    """
    Data class representing the result of HNSW queries with a specific ef parameter.

    Attributes:
        ef: The size of the dynamic list for the nearest neighbors (used during search).
        avg_recall: The average recall rate of a query.
        avg_query_time: The average time taken to execute a query.
        queries_per_second: The number of queries processed per second.
        total_time: The total time taken to execute all queries.
        num_queries: The total number of queries executed.
        k: The number of nearest neighbors considered.
    """
    ef: int
    avg_recall: float
    avg_query_time: float
    queries_per_second: float
    total_time: float
    num_queries: int
    k: int


@dataclass(frozen=True)
class HNSWQueryModeResult:
    """
    Data class representing the results of HNSW queries for a specific query mode.

    Attributes:
        mode: The query mode used (see :class:`QueryMode`).
        ef_results: A list of results for different ef values (see :class:`HNSWQueryEFResult`).
    """
    mode: QueryMode
    ef_results: list[HNSWQueryEFResult]


@dataclass(frozen=True)
class HNSWQueryRunnerResult:
    """
    Data class representing the combined results of HNSW queries for multiple query modes.

    Attributes:
        mode_results: A list of query mode results (see :class:`HNSWQueryModeResult`).
    """
    mode_results: list[HNSWQueryModeResult]


@dataclass(frozen=True)
class HNSWRunnerResult:
    """
    Data class representing the overall result of an HNSW task.

    Attributes:
        client: The used database client (see :class:`BaseClient`).
        index_config: The configuration of the HNSW index (see :class:`BaseHNSWConfig`).
        insert_result: The result of the insertion operation (see :class:`InsertRunnerResult`).
        query_result: The result of the query operations (see :class:`HNSWQueryRunnerResult`).
        index_size: The size of the HNSW index in MB.
        disk_size: The disk size used by the database in MB.
    """
    client: BaseClient
    index_config: BaseHNSWConfig
    insert_result: InsertRunnerResult
    query_result: HNSWQueryRunnerResult
    index_size: float
    disk_size: float
