import numpy as np
import matplotlib.pyplot as plt
from ya_glm.linalg_utils import euclid_norm


def plot_successive_diffs(values, norm='MAD', log=True,
                          marker='.', color='black', **kws):
    """
    Gets the successive differences between values.

    Parameters
    ----------
    values: list of array-like
        The list of values.

    norm: str
        Which norm to use of the differences. See get_successive_diffs()

    log: bool
        Whether or not to log the differences.

    **kws:
        Keyword arguments to plt.plot()
    """
    diffs = get_successive_diffs(values=values, norm=norm)
    if log:
        diffs = np.log10(diffs)
    plt.plot(diffs, marker=marker, color=color, **kws)

    ylab = '{} successive difference'.format(norm)
    if log:
        ylab = 'log10({})'.format(ylab)
    plt.ylabel(ylab)


def get_successive_diffs(values, norm='MAD'):
    """
    Gets the successive differences between values.

    Parameters
    ----------
    values: list of array-like
        The list of values.

    norm: str
        Which norm to use of the differences. Must be one of ['L1', 'L2', 'RMSE', 'MAD', 'MAD'].

    Output
    ------
    diffs: list of floats
        The differences.
    """

    norm = norm.lower()

    if norm == 'l2':
        norm = euclid_norm

    if norm == 'rmse':
        norm = lambda x: (np.array(x).reshape(-1) **2).mean()

    elif norm == 'l1':
        norm = lambda x: abs(np.array(x).reshape(-1)).sum()

    elif norm == 'mad':
        norm = lambda x: abs(np.array(x).reshape(-1)).mean()

    elif norm == 'max':
        norm = lambda x: abs(np.array(x)).max().mean()

    else:
        raise ValueError("Bad input for norm {}".format(norm))

    n_values = len(values)
    return np.array([norm(values[i + 1] - values[i])
                    for i in range(n_values - 1)])
