from typing import Optional

from ..client.base_client import BaseClient


class MockChromaClient(BaseClient):

    def __init__(self):
        MockChromaClient.__name__ = "ChromaClient"

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        pass

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        pass

    def create_index(self) -> None:
        pass

    def disk_storage(self) -> float:
        pass

    def index_storage(self) -> float:
        pass

    def load(self) -> None:
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        pass

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        pass

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        pass


class MockMilvusClient(BaseClient):

    def __init__(self):
        MockMilvusClient.__name__ = "MilvusClient"

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        pass

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        pass

    def create_index(self) -> None:
        pass

    def disk_storage(self) -> float:
        pass

    def index_storage(self) -> float:
        pass

    def load(self) -> None:
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        pass

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        pass

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        pass


class MockRedisClient(BaseClient):

    def __init__(self):
        MockRedisClient.__name__ = "RedisClient"

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        pass

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        pass

    def create_index(self) -> None:
        pass

    def disk_storage(self) -> float:
        pass

    def index_storage(self) -> float:
        pass

    def load(self) -> None:
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        pass

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        pass

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        pass


class MockPgvectorClient(BaseClient):

    def __init__(self):
        MockPgvectorClient.__name__ = "PgvectorClient"

    def insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None, start_id: int = 0) -> None:
        pass

    def batch_insert(self, embeddings: list[list[float]], metadata: Optional[list[str]] = None,
                     start_id: int = 0) -> None:
        pass

    def create_index(self) -> None:
        pass

    def disk_storage(self) -> float:
        pass

    def index_storage(self) -> float:
        pass

    def load(self) -> None:
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        pass

    def filtered_query(self, query: list[float], k: int, keyword_filter: str) -> list[int]:
        pass

    def ranged_query(self, query: list[float], k: int, distance: float) -> list[int]:
        pass


client_mock_mapper = {
    "ChromaClient": MockChromaClient,
    "MilvusClient": MockMilvusClient,
    "RedisClient": MockRedisClient,
    "PgvectorClient": MockPgvectorClient
}
