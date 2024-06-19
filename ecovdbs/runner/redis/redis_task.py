from ..base.runner_config import HNSWCase, HNSWTask
from ...client.redis.redis_config import RedisHNSWConfig
from ...client.redis.redis_client import RedisClient
from ...dataset.dataset import Dataset
from dataclasses import dataclass


@dataclass
class RedisHNSWTask(HNSWTask):
    """
    Represents a task for running HNSW (Hierarchical Navigable Small World) on a Redis database.

    Attributes:
        index_config: Configuration for the HNSW index in Redis.
        client: Redis client to interact with the Redis database.
        dataset: Dataset to be used for the HNSW task.
        ef_search: A list of sizes for the dynamic list for the nearest neighbors (used during search).
    """
    index_config: RedisHNSWConfig
    client: RedisClient
    dataset: Dataset
    ef_search: list[int]

    def __init__(self, case: HNSWCase):
        """
        Initialize the RedisHNSWTask with a given HNSW case configuration.

        :param case: The HNSW case configuration (see :class:`HNSWCase`).
        """
        self.index_config = RedisHNSWConfig(metric_type=case.dataset.metric_type, M=case.hnsw_config.M,
                                            ef_construction=case.hnsw_config.ef_construction)
        self.client = RedisClient(dimension=case.dataset.dimension, index_config=self.index_config)
        self.dataset = case.dataset
        self.ef_search = case.hnsw_config.ef_search
