import numpy as np
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager


def create_shared_ndarray(arr: np.ndarray, name: str = None, smn: SharedMemoryManager = None) -> tuple:
    """
    Creates a shared ndarray and returns it.
    1. Creates shared memory buffer
    2. Creates np.ndarray backed by shared memory buffer
    3. Copies data to shared np.ndarray
    4. Returns (shared_ndarray, shared_memory_used_as_buffer)

    NOTE: for a arrays of size 0, it will create an buffer that still has the size of one element of such array

    @smn: potentially shared memory manager  (will ignore name argument!)
    """
    size = arr.nbytes

    if size == 0:
        # fake arr
        size = np.zeros((1, ), dtype=arr.dtype).nbytes

    if smn is not None:
        shm = smn.SharedMemory(size=size)
    else:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]

    return shared_arr, shm


def get_shared_ndarray(attr: tuple) -> tuple:
    """
    Returns a np.ndarray backed by the shared buffer with given name.
    """
    name, shape, dtype = attr  # name: str, shape: tuple, dtype: str

    existing_shm = shared_memory.SharedMemory(name=name)
    return np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf), existing_shm


def combine_variable_sized_arrays_to_one(nested_arr: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a list of np.ndarray's where each may be of variable length but has the same lower-dimensionality, e.g., shapes [(3, 4, 3), (6, 4, 3), (1, 4, 3)],
        returns a single np.ndarray with the arrays stacked, as well as a an np.ndarray of offsets and one with lengths to reconstruct the array values. 
    """
    sizes = np.array([len(arr) for arr in nested_arr])
    offsets = np.insert(np.cumsum(sizes[:-1]), 0, 0).astype(int)

    stacked = np.vstack([arr for arr in nested_arr])

    return stacked, offsets, sizes


def get_arr_at_idx_from_combined_array(combined_arr: np.ndarray, offsets: list, sizes: list, idx: int, copy: bool = False) -> np.ndarray:
    start = offsets[idx]
    end = start + sizes[idx]
    ret = combined_arr[start:end]

    if copy:
        return ret.copy()
    else:
        return ret


def uncombine_variable_sized_arrays_from_one(combined_arr: np.ndarray, offsets: list, sizes: list, copy: bool = False) -> list[np.ndarray]:
    return [get_arr_at_idx_from_combined_array(combined_arr=combined_arr, offsets=offsets, sizes=sizes, idx=i, copy=copy) for i in range(len(offsets))]
