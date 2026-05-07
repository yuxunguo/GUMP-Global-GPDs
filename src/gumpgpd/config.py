from joblib import Memory
import numpy as np
import functools
import hashlib
import os
import shutil
from diskcache import FanoutCache


# Turn it on (set to true) so the cost function will save the results to files
# Only used after the fit is finished!
Export_Mode = False

INC_JPSI = False

INC_gGFF = False

_log_interval_minutes = 10  # Interval in minutes for logging time during minimization

NC = 3
CF = (NC**2 - 1) / (2 * NC)
CA = NC
CG = CF - CA/2
TF = 0.5

# Use sharded on-disk cache to reduce writer contention in multiprocessing workloads.
cachedir = '__cachedir__'
cache_on_disk = FanoutCache(cachedir, shards=128,timeout=0.5)
memory = Memory(location=cachedir, verbose=0, compress=3)
memoryram = Memory(location=None, verbose=0)


def clear_cachedir():
    """Clear all on-disk cache data under `cachedir` and recreate the folder."""
    cache_on_disk.clear()
    memory.clear(warn=False)
    shutil.rmtree(cachedir, ignore_errors=True)
    os.makedirs(cachedir, exist_ok=True)
    print(f"Cleared on-disk cache at '{cachedir}' and recreated the folder.")

def Hybrid_Cache(func):
    """
    Hybrid cache: RAM (in-process) + disk (diskcache)
    - First argument can be a NumPy array or any hashable scalar (int, float, complex, str, tuple)
    - Remaining arguments must be hashable
    """
    ram_cache = {}

    def serialize_first_arg(arg):
        """
        Generate a robust key for the first argument:
        - NumPy array: hash via SHA256
        - Scalars: use directly
        """
        if isinstance(arg, np.ndarray):
            arr_bytes = np.ascontiguousarray(arg).view(np.uint8)
            h = hashlib.sha256(arr_bytes).hexdigest()
            return (h, arg.shape, str(arg.dtype))
        else:
            return arg

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Handle first argument (positional or keyword)
        if args:
            first_arg = args[0]
            rest_args = args[1:]
        else:
            raise TypeError("Missing required first argument for caching")

        # Build composite key for caching
        key = (func.__module__,
            func.__name__,
            serialize_first_arg(first_arg),
            rest_args,
            tuple(sorted(kwargs.items())))

        # Check RAM cache first
        if key in ram_cache:
            return ram_cache[key]

        # Check disk cache
        if key in cache_on_disk:
            result = cache_on_disk[key]
        else:
            # Compute result and store in disk cache
            result = func(first_arg, *rest_args, **kwargs)
            cache_on_disk[key] = result

        # Store in RAM cache for faster access
        ram_cache[key] = result
        return result

    return wrapper
