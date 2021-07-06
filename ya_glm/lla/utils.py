import numpy as np
from numbers import Number


def safe_concat(a, b):
    """
    Concatenates two values into a one-dimensional array. Works if a or b is a number or matrix.

    Parameters
    ----------
    a, b: array-like, Number
        The two values to concatenate.

    Output
    ------
    cat: np.array
        The concatenated values.
    """

    to_cat = []
    if isinstance(a, Number):
        to_cat.append([a])
    elif len(a) == 1:
        to_cat.append([float(a)])
    else:
        to_cat.append(np.array(a).reshape(-1))

    if isinstance(b, Number):
        to_cat.append([b])
    elif len(b) == 1:
        to_cat.append([float(b)])
    else:
        to_cat.append(np.array(b).reshape(-1))

    return np.concatenate(to_cat)
