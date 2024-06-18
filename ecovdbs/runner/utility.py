import time
from functools import wraps


def time_it(func):
    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return time_it_wrapper
