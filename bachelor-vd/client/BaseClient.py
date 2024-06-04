from abc import abstractmethod, ABC


class BaseClient(ABC):

    @abstractmethod
    def __init__(self, db_config: dict, drop_old: bool):
        raise NotImplementedError

    @abstractmethod
    def init_db(self):
        raise NotImplementedError

    @abstractmethod
    def crate_index(self):
        raise NotImplementedError

    @abstractmethod
    def insert(self, embeddings: list[list[float]]):
        raise NotImplementedError

    @abstractmethod
    def query(self, query: list[float], k: int):
        raise NotImplementedError
