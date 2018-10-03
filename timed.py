import time

from functools import wraps

LAST_TIMED = dict()


def timed(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        LAST_TIMED[func.__name__] = end - start
        return results
    return new_func