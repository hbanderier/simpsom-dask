import subprocess
import multiprocessing
try:
    from dask.diagnostics import ProgressBar
except ModuleNotFoundError:
    pass
import os
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
    
    
def _compute(obj, progress: bool = False, **kwargs):
    kwargs = COMPUTE_KWARGS | kwargs
    try:
        if progress:
            with ProgressBar():
                return obj.compute(**kwargs)
        else:
            return obj.compute(**kwargs)
    except AttributeError:
        return obj

    