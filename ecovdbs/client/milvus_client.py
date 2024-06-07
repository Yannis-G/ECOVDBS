import pymilvus
from pymilvus import DataType, connections, FieldSchema, CollectionSchema, Collection, utility

from .base_client import BaseClient
from .milvus_config import MilvusConfig


class MilvusClient(BaseClient):
    def __init__(self, dimension: int, db_config=None):
        if db_config is None:
            db_config = MilvusConfig().to_dict()
        self.__dimension = dimension
        self.__db_config = db_config
        self.__collection_name = "ecovdbs"
        connections.connect(uri=f"http://{self.__db_config['host']}:{self.__db_config['port']}")
        if utility.has_collection(self.__collection_name):
            utility.drop_collection(self.__collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.__dimension),
        ]
        schema = CollectionSchema(fields)
        self.__collection = Collection(self.__collection_name, schema)

    def insert(self, embeddings: list[list[float]]) -> None:
        embeddings = [{"id": i, "vector": v} for i, v in enumerate(embeddings)]
        self.__collection.insert(data=embeddings)

    def batch_insert(self, embeddings: list[list[float]]) -> None:
        pass

    def crate_index(self) -> None:
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.__collection.create_index("vector", index)

    def disk_storage(self):
        pass

    def index_storage(self):
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        self.__collection.load()
        res = self.__collection.search(data=[query], anns_field="vector", param=search_params, limit=k)
        return [x.id for x in res[0]]
