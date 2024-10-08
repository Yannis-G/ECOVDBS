import json
import os
import time
from dataclasses import is_dataclass, fields
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from .case_config import QueryMode
from .chroma.chroma_task import ChromaHNSWTask
from .milvus.milvus_task import MilvusHNSWTask
from .redis.redis_task import RedisHNSWTask
from .pgvector.pgvector_task import PgvectorHNSWTask
from .result_config import HNSWRunnerResult
from .mock_clients import client_mock_mapper
from ..client.base_client import BaseClient
from ..client.base_config import BaseHNSWConfig
from ..config import RESULT_BASE_PATH

client_mapper = {
    "CHROMA": ChromaHNSWTask,
    "MILVUS": MilvusHNSWTask,
    "REDIS": RedisHNSWTask,
    "PGVECTOR": PgvectorHNSWTask
}


def time_it(func) -> Callable[..., tuple[Any, float]]:
    """
    A decorator that measures the execution time of a function.

    :param func: The function to be wrapped and timed.
    :return: A wrapped function that returns a tuple containing the original function's result and the execution time in
        seconds.
    """

    @wraps(func)
    def time_it_wrapper(*args, **kwargs) -> tuple[Any, float]:
        """
        Wrapper function that measures the execution time of the wrapped function.

        :param args: Positional arguments to pass to the wrapped function.
        :param kwargs: Keyword arguments to pass to the wrapped function.
        :return: A tuple containing the result of the wrapped function and the execution time in seconds.
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return time_it_wrapper


def dataclass_to_dict(obj: Any) -> Any:
    """
    Convert a dataclass object (and its nested attributes) into a dictionary.

    :param obj: The object to convert. It can be a dataclass instance, list, tuple, dict, or another object.
    :return: Equivalent dictionary representation of the object.
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)
            if isinstance(value, BaseClient):
                result[key] = type(value).__name__
            elif isinstance(value, BaseHNSWConfig):
                result[key] = {"index_param": value.index_param(), "search_param": value.search_param()}
            elif isinstance(value, Enum):
                result[key] = value.name
            else:
                result[key] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj


def save_hnsw_runner_result(result: HNSWRunnerResult, path: Optional[str] = None) -> None:
    """
    Save an HNSWRunnerResult object to a JSON file.

    :param result: HNSWRunnerResult object to save.
    :param path: Path to save the JSON file.
    """
    if path is None:
        path = os.path.join(RESULT_BASE_PATH,
                            f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{type(result.client).__name__}-result.json")
    with open(path, 'w') as file:
        json.dump(dataclass_to_dict(result), file, indent=4)


class MockBaseHNSWConfig(BaseHNSWConfig):

    def __init__(self, index_param: dict[str, Any], search_param: dict[str, Any]) -> None:
        self.index_param = index_param
        self.search_param = search_param

    def index_param(self) -> dict[str, Any]:
        pass

    def search_param(self) -> dict[str, Any]:
        pass

    def change_ef_search(self, ef: int) -> None:
        pass


def dict_to_dataclass(data: Any, cls: Any) -> Any:
    if cls == BaseClient:
        return client_mock_mapper[data]()
    elif cls == BaseHNSWConfig:
        return MockBaseHNSWConfig(data["index_param"], data["search_param"])
    elif cls == QueryMode:
        return cls[data]
    elif isinstance(data, dict):
        fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        return cls(**{k: dict_to_dataclass(v, fieldtypes[k]) for k, v in data.items()})
    elif isinstance(data, list):
        return [dict_to_dataclass(item, cls.__args__[0]) for item in data]
    return data


def read_hnsw_runner_result(path: str) -> HNSWRunnerResult:
    with open(path, 'r') as file:
        data = json.load(file)
    return dict_to_dataclass(data, HNSWRunnerResult)
