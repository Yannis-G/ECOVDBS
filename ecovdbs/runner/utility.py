import json
import time
from dataclasses import asdict, is_dataclass, fields
from enum import Enum
from functools import wraps
from typing import Any, Callable

from .result_config import HNSWRunnerResult
from ..client.base_client import BaseClient
from ..client.base_config import BaseHNSWConfig


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


def save_hnsw_runner_result(path: str, result: HNSWRunnerResult) -> None:
    with open(path, 'w') as file:
        json.dump(dataclass_to_dict(result), file, indent=4)
