import subprocess
import multiprocessing
try:
    from dask.diagnostics import ProgressBar
except ModuleNotFoundError:
    pass
import os
import numpy as np
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
MEMORY_LIMIT = int(os.environ.get("SLURM_MEM_PER_NODE", "150000")) // N_WORKERS
COMPUTE_KWARGS = {
    "processes": True,
    "threads_per_worker": 1,
    "n_workers": N_WORKERS,
    "memory_limit": MEMORY_LIMIT,
}


def find_max_cuda_threads():
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        n_smp = dev.attributes["MultiProcessorCount"]
        max_thread_per_smp = dev.attributes["MaxThreadsPerMultiProcessor"]
        return n_smp * max_thread_per_smp
    except:
        print("Cupy is not available.")
        return 0


def find_cpu_cores():
    try:
        return multiprocessing.cpu_count()
    except:
        print("Could not infer #CPU_cores")
        return 0
    
    
def _get(array):
    try:
        return array.get()
    except AttributeError:
        return array
    
    
def compute(obj, progress_flag: bool = False, **kwargs):
    kwargs = COMPUTE_KWARGS | kwargs
    try:
        client # in globals
    except NameError:
        try:
            if progress_flag:
                with ProgressBar():
                    return obj.compute(**kwargs)
            else:
                return obj.compute(**kwargs)
        except AttributeError:
            return obj
    try:
        if progress_flag:
            obj = client.gather(client.persist(obj))
            progress(obj)
            return obj
        else:
            return client.compute(obj)
    except AttributeError:
        return obj
    
    
def triangulize(span: np.ndarray):
    filter_ = (span > 0).astype(int)
    out = span * 2 + 1
    out = out - 2 * filter_
    out = out * np.asarray([1, -1])[filter_]
    return out


def normalize(X):
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)
    X = (X - meanX[None, :]) / stdX[None, :]
    return X, meanX, stdX


def revert_normalize(X, meanX, stdX):
    X = X * stdX[None, :] + meanX[None, :]
    return X

    