import chromadb
from chromadb import ClientAPI, Collection, QueryResult
from chromadb.utils.batch_utils import create_batches

from .base_client import BaseClient, BaseIndexConfig, BaseConfig
from .chroma_config import ChromaConfig, ChromaIndexConfig


class ChromaClient(BaseClient):
    """
    A client for interacting with a Chroma database (see https://docs.trychroma.com/). Interface is the same as
    :class:`BaseClient`.
    """

    def __init__(self, dimension: int, index_config: BaseIndexConfig = ChromaIndexConfig(),
                 db_config: BaseConfig = ChromaConfig()) -> None:
        """
        Initialize the ChromaClient with a given database configuration.

        :param dimension: Not relevant! The first inserted vector decides the dimension of the collection.
        :param index_config: Configuration for the index (see :class:`ChromaIndexConfig`).
        :param db_config: Configuration for the database connection (see :class:`ChromaConfig`).
        """
        self.__dimension: int = dimension
        self.__index_config: BaseIndexConfig = index_config
        self.__db_config: dict = db_config.to_dict()
        self.__collection_name: str = "ecovdbs"
        self.__client: ClientAPI = chromadb.HttpClient(host=self.__db_config["host"], port=self.__db_config["port"])

        # Ensure the client is alive by checking the heartbeat.
        assert self.__client.heartbeat() is not None

        # Try to delete the collection if it already exists.
        try:
            # Empties and completely resets the database. ⚠️
            self.__client.reset()
            self.__client.delete_collection(self.__collection_name)
        except Exception:
            pass

        # Get or create the collection.
        self.__collection: Collection = self.__client.get_or_create_collection(name=self.__collection_name,
                                                                               metadata=self.__index_config.index_param())

    def insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        # self.__client.max_batch_size >> 41666
        ids: list[str] = [str(i) for i in range(start_id, start_id + len(embeddings))]
        self.__collection.add(ids=ids, embeddings=embeddings)

    def batch_insert(self, embeddings: list[list[float]], start_id: int = 0) -> None:
        ids: list[str] = [str(i) for i in range(start_id, start_id + len(embeddings))]
        batches = create_batches(api=self.__client, ids=ids, embeddings=embeddings)
        for batch in batches:
            self.__collection.add(ids=batch[0], embeddings=batch[1])

    def create_index(self):
        """
        Not implemented! Chroma DB automatically creates an index of the embeddings as they are inserted into the
        collection. (https://github.com/zylon-ai/private-gpt/discussions/563)
        """
        pass

    def disk_storage(self):
        # https://cookbook.chromadb.dev/core/storage-layout/: The persistence directory consists of two files: the
        # `chroma.sqlite3` and the folder named after the collections UUID for the HNSW index. As I understand it,
        # only metadata is stored in the SQLite, and the index along with all vectors is stored in the directory.
        # Therefore, it is not possible to determine how much memory the index consumes versus how much memory the
        # vectors consume. https://cookbook.chromadb.dev/core/resources/: According to the formula, one needs n * d *
        # 4 bytes of RAM for the index. The disk space depends on the metadata and the number of vectors. According
        # to heuristics, 2-4 * RAM is needed for the HNSW index.
        # https://cookbook.chromadb.dev/core/concepts/#vector-index-hnsw-index: The index is stored in a subdir of
        # your persistent dir, named after the collection id (UUID-based).
        pass

    def index_storage(self):
        # See disk_storage
        pass

    def query(self, query: list[float], k: int) -> list[int]:
        """
        Query the database with a given embedding and return the top k results.

        :param query: The query embedding.
        :param k: The number of results to return.
        :return: The id of the top k results from the query.
        """
        res: QueryResult = self.__collection.query(query_embeddings=query, n_results=k)
        return [int(id) for id in res["ids"][0]]
