import chromadb

from BaseClient import BaseClient
from ChromaConfig import ChromaConfig


class ChromaClient(BaseClient):

    def __init__(self, db_config: dict = ChromaConfig().to_dict(), drop_old: bool = False):
        self.__db_config = db_config
        self.__collection_name = "bvd"
        client = chromadb.HttpClient(host=self.__db_config["host"], port=self.__db_config["port"])
        assert client.heartbeat() is not None
        if drop_old:
            try:
                client.delete_collection(self.__collection_name)
            except Exception:
                pass

    def init_db(self):
        self._client = chromadb.HttpClient(host=self.__db_config["host"], port=self.__db_config["port"])
        self._collection = self._client.get_or_create_collection(name="bvd")

    def crate_index(self):
        pass

    def insert(self, embeddings: list[list[float]]):
        ids = [str(i) for i, _ in enumerate(embeddings)]
        self._collection.upsert(ids=ids, embeddings=embeddings)

    def query(self, query: list[float], k: int):
        return self._collection.query(query_embeddings=query, n_results=k)


if __name__ == '__main__':
    client = ChromaClient(drop_old=True)
    client.init_db()
    client.insert([[1, 2, 3], [2, 3, 4]])
    print(client.query([17, 5, 9], 2))
