import time
from functools import wraps

def timed(attribute, key):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter_ns()
            result = function(self, *args, **kwargs)
            getattr(self, attribute)[key] += time.perf_counter_ns() - t0 
            return result
        return wrapper
    return decorator
