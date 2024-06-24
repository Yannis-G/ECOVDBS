from dataclasses import dataclass

from ..case_config import HNSWCase
from ..task_config import HNSWTask, InsertConfig, HNSWQueryConfig, IndexTime
from ...client.chroma.chroma_client import ChromaClient
from ...client.chroma.chroma_config import ChromaHNSWConfig
from ...dataset.dataset import Dataset


@dataclass
class ChromaHNSWTask(HNSWTask):
    """
    Represents a task for running HNSW (Hierarchical Navigable Small World) on a Chroma database.

    Attributes:
        client: Chroma client to interact with the Chroma database.
        dataset: Dataset to be used for the HNSW task.
        insert_config: Configuration for the insertion operation.
        query_config: Configuration for the query operation.
    """
    client: ChromaClient
    dataset: Dataset
    insert_config: InsertConfig
    query_config: HNSWQueryConfig

    def __init__(self, case: HNSWCase):
        """
        Initialize the ChromaHNSWTask with a given HNSW case configuration.

        :param case: The HNSW case configuration (see :class:`HNSWCase`).
        """
        index_config = ChromaHNSWConfig(metric_type=case.dataset.metric_type, M=case.hnsw_config.M,
                                        construction_ef=case.hnsw_config.ef_construction)
        self.client = ChromaClient(dimension=case.dataset.dimension, index_config=index_config)
        self.dataset = case.dataset
        index_time = case.index_time if case.index_time is IndexTime.NO_INDEX else IndexTime.NO_INDEX
        self.insert_config = InsertConfig(index_time=index_time, query_modes=case.query_modes)
        self.query_config = HNSWQueryConfig(ef_search=case.hnsw_config.ef_search, index_config=index_config,
                                            query_modes=case.query_modes)
