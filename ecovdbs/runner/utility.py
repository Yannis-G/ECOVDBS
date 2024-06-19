import time
from functools import wraps
from typing import Any, Callable


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
