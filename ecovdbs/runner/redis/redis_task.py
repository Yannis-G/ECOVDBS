from ..base.runner_config import HNSWCase, HNSWTask
from ...client.redis.redis_config import RedisHNSWConfig
from ...client.redis.redis_client import RedisClient
from ...dataset.dataset import Dataset
from dataclasses import dataclass


@dataclass
class RedisHNSWTask(HNSWTask):
    index_config: RedisHNSWConfig
    client: RedisClient
    dataset: Dataset

    def __init__(self, case: HNSWCase):
        self.index_config = RedisHNSWConfig(metric_type=case.dataset.metric_type, M=case.hnsw_config.M,
                                            ef_construction=case.hnsw_config.ef_construction)
        self.client = RedisClient(dimension=case.dataset.dimension, index_config=self.index_config)
        self.dataset = case.dataset
