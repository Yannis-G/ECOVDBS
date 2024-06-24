from dataclasses import dataclass

from ..case_config import HNSWCase
from ..task_config import HNSWTask, InsertConfig, HNSWQueryConfig, IndexTime
from ...client.pgvector.pgvector_client import PgvectorClient
from ...client.pgvector.pgvector_config import PgvectorHNSWConfig
from ...dataset.dataset import Dataset


@dataclass
class PgvectorHNSWTask(HNSWTask):
    """
    Represents a task for running HNSW (Hierarchical Navigable Small World) on a Pgvector database.

    Attributes:
        client: Pgvector client to interact with the Pgvector database.
        dataset: Dataset to be used for the HNSW task.
        insert_config: Configuration for the insertion operation.
        query_config: Configuration for the query operation.
    """
    client: PgvectorClient
    dataset: Dataset
    insert_config: InsertConfig
    query_config: HNSWQueryConfig

    def __init__(self, case: HNSWCase):
        """
        Initialize the PgvectorHNSWTask with a given HNSW case configuration.

        :param case: The HNSW case configuration (see :class:`HNSWCase`).
        """
        index_config = PgvectorHNSWConfig(metric_type=case.dataset.metric_type, m=case.hnsw_config.M,
                                          ef_construction=case.hnsw_config.ef_construction)
        self.client = PgvectorClient(dimension=case.dataset.dimension, index_config=index_config)
        self.dataset = case.dataset
        index_time = case.index_time if case.index_time is not IndexTime.NO_INDEX else IndexTime.PRE_INDEX
        self.insert_config = InsertConfig(index_time=index_time, query_modes=case.query_modes)
        self.query_config = HNSWQueryConfig(ef_search=case.hnsw_config.ef_search, index_config=index_config,
                                            query_modes=case.query_modes)
