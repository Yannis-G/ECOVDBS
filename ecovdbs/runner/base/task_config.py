from dataclasses import dataclass

from .case_config import IndexTime, QueryMode
from ...client.base.base_client import BaseClient
from ...client.base.base_config import BaseHNSWConfig
from ...dataset.dataset import Dataset


@dataclass(frozen=True)
class InsertConfig:
    """
    Configuration class for the insertion operation.

    Attributes:
        index_time: The time at which the index is created (see :class:`IndexTime`).
        query_modes: The query modes (see :class:`QueryMode`).
    """
    index_time: IndexTime
    query_modes: list[QueryMode]


@dataclass(frozen=True)
class HNSWQueryConfig:
    """
    Configuration class for the query operation.

    Attributes:
        ef_search: A list of sizes for the dynamic list for the nearest neighbors (used during search).
        index_config: Configuration for the HNSW index (see :class:`BaseHNSWConfig`).
        query_modes: The query modes (see :class:`QueryMode`).
    """
    ef_search: list[int]
    index_config: BaseHNSWConfig
    query_modes: list[QueryMode]


@dataclass(init=False)
class HNSWTask:
    """
    Task class for HNSW operations, including configuration, client, and dataset.

    Attributes:
        client: The client to interact with the HNSW index (see :class:`BaseClient`).
        dataset: The dataset to be used for the task (see :class:`Dataset`).
        insert_config: Configuration for the insertion operation (see :class:`InsertConfig`).
        query_config: Configuration for the query operation (see :class:`QueryConfig`).
    """
    client: BaseClient
    dataset: Dataset
    insert_config: InsertConfig
    query_config: HNSWQueryConfig
