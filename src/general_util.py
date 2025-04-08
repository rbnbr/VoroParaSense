import numpy as np
import pandas as pd


def get_samples_and_labels_for(df: pd.DataFrame, keys: list, label_key: str, label_names_key: str = None, remove_nans: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the samples and labels for the given keys from a pd.Dataframe.
    @label_key: key that defines the labels.
    @remove_nans: if True, removes the nan rows.
    """
    df_ = df[keys]
    values = df_.values
    labels = df[label_key].values

    if label_names_key is not None:
        label_names = df[label_names_key].values

    if remove_nans:
        not_na = df_.notna()[keys].values
        selector = np.all(not_na, 1)
        values = values[selector]
        labels = labels[selector]

        if label_names_key is not None:
            label_names = label_names[selector]

    if label_names_key is None: 
        return values, labels
    return values, labels, label_names


def get_continuous_occurrences_of_true_and_false(arr: np.ndarray) -> np.ndarray:
    """
    Returns an integer array that counts the continuous occurrences of True and False in the given boolean array.
    E.g., array([ True,  True,  True, False,  True,  True,  True, False,  True,
       False])  => array([3, 1, 3, 1, 1, 1])  (3 True, 1 False, 3 True, 1 False, 1 True, 1 False).
    To filter for True's: get every second element starting with the first.  [:][::2]
    To filter for False's: get every second element starting with the second.  [1:][::2]
    """
    assert arr.dtype == bool, "dtype must be bool"
    assert len(arr.shape) == 1, "shape must be length 1, i.e., a one dimensional array"

    ret = np.diff(np.where(np.concatenate(([arr[0]], arr[:-1] != arr[1:], [True])))[0])

    if len(arr) > 0 and not arr[0]:
        # handle special case where the array starts with False
        init_falses = 0
        for i in range(len(arr)):
            if not arr[i]:
                init_falses += 1
                continue
            break

        ret = np.concat([[0, init_falses], ret])

    return ret


def create_discrete_values_mapping(df: pd.DataFrame, sort: bool = True) -> dict:
    """
    Create a discrete value mapping for all columns in df that have dtype=object.
    @return {
        col_name: {
            discrete_value_1: 0,
            discrete_value_2: 1,
            ...
        },
        ...
    }
    @sort: if True, sorts them, otherwise, assigns values according to order of appearance. 
    """
    ret = dict()
    for col, dtype in enumerate(df.dtypes):
        if dtype == object:
            m = dict()
            col_name = df.columns[col]
            values = df.iloc[:, col]

            if sort:
                values = np.sort(values)

            idx = 0
            for value in values:
                if value not in m:
                    m[value] = idx
                    idx += 1
            
            ret[col_name] = m

    return ret


def apply_discrete_values_mapping(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    """
    Returns a copy of df with all discrete values replaced by the values specified in the provided mapping m.
    m should look like the output of create_discrete_values_mapping.
    """
    vals = []
    for col_name in df.columns:
        if col_name in m:
            # apply mapping
            m_col = m[col_name]

            col_vals = [m_col[v] for v in df[col_name].values]
        else:
            col_vals = df[col_name].values.copy()  # no mapping

        vals.append(col_vals)

    df_ = pd.DataFrame(columns=df.columns.copy(), index=df.index.copy())
    for i, col_name in enumerate(df.columns):
        df_[col_name] = vals[i]
    
    return df_


def rotation_matrix_3d(radians: float, dim: int) -> np.ndarray:
    """
    Returns a rotation matrix to rotate a 3D vector around the specified axis by the specified radians.
    Can be applied to rotate a set of points in Matrix M via M_rotated = M @ R
    """
    R = np.eye(3)
    cr = np.cos(radians)
    sr = np.sin(radians)

    if dim == 0:
        R[1, 1] = cr
        R[1, 2] = -sr
        R[2, 1] = sr
        R[2, 2] = cr
    elif dim == 1:
        R[0, 0] = cr
        R[0, 2] = sr
        R[2, 0] = -sr
        R[2, 2] = cr
    elif dim == 2:
        R[0, 0] = cr
        R[0, 1] = -sr
        R[1, 0] = sr
        R[1, 1] = cr
    else:
        raise RuntimeError(f"invalid dim, must be 0, 1, or 2. got {dim}")

    return R

