import numpy as np
import numba


def make_object_array(arr: list, dtype=object) -> np.ndarray:
    """
    Given any list, returns an np.ndarray with given dtype and shape (len(arr), ) that has the non-converted elements as entries.
    This avoids problems when initializing np.array with dtype=object where the sub-arrays sometimes inherit the dtype=object even though they should each have their own dtype.
    """
    ret = np.empty((len(arr), ), dtype=dtype)
    for i in range(len(arr)):
        ret[i] = arr[i]
    
    return ret


def normalize_vec_ret_l(v: np.ndarray, return_length: bool = False) -> tuple:
    l = np.linalg.norm(v)
    if not return_length:
        return v / l
    return v / l, l


@numba.njit()
def normalize_vec(v: np.ndarray) -> np.ndarray:
    l = np.linalg.norm(v)
    return v / l


@numba.njit()
def normalize_vec_full(v: np.ndarray) -> tuple:
    l = np.linalg.norm(v)
    return v / l, l


def is_close(a: np.ndarray, b: np.ndarray, a_eps: float = 1e-12) -> bool:
    return np.all(np.abs(a-b) < a_eps)


@numba.njit()
def is_close2(a, b):
    for i in range(len(a)):
        if abs(a[i] - b[i]) > 1e-12:
            return False
    return True


@numba.njit()
def all_zero(a):
    for i in range(len(a)):
        if a[i] > 1e-12:
            return False
    return True


@numba.njit()
def nb_vstack(l: list) -> np.ndarray:
    assert len(l) > 0

    s = sum([len(e) for e in l])

    shape = (s, *l[0].shape[1:])

    arr = np.empty(shape=shape, dtype=l[0].dtype)

    p = 0

    for i in range(len(l)):
        arr_ = l[i]
        arr[p:p+len(arr_)] = arr_[:]
        p += len(arr_)

    return arr


@numba.njit()
def nb_stack(l: list) -> np.ndarray:
    assert len(l) > 0

    shape = (len(l), *l[0].shape)

    arr = np.empty(shape=shape, dtype=l[0].dtype)

    for i in range(len(l)):
        arr[i] = l[i]

    return arr


@numba.njit()
def nb_mean0(arr: np.ndarray) -> np.ndarray:
    return np.array([arr[..., i].mean() for i in range(arr.shape[-1])])


@numba.njit()
def nb_min0(arr: np.ndarray) -> np.ndarray:
    return np.array([arr[..., i].min() for i in range(arr.shape[-1])])


@numba.njit()
def nb_max0(arr: np.ndarray) -> np.ndarray:
    return np.array([arr[..., i].max() for i in range(arr.shape[-1])])


@numba.njit()
def nb_linalg_norm_1(arr: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(arr[i]) for i in range(arr.shape[0])])


@numba.njit()
def nb_sum0(arr: np.ndarray) -> np.ndarray:
    return np.array([arr[..., i].sum() for i in range(arr.shape[-1])])


# src: https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks_of_size_n(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# src: https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def n_chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
