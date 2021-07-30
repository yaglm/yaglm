from scipy.sparse.linalg import svds
import numpy as np


def smallest_sval(X, solver='lobpcg', **kws):
    """
    Computes the smallest singular value of a matrix using
    scipy.sparse.linalg.svds

    Parameters
    ----------
    X: array-like

    solver: str
        Which solver to use. Must be one of ['lobpcg', 'arpack']

    **kws
        Kws for svds

    Output
    ------
    smallest_sval: float
        The smallest singular value of X
    """

    # for 1d arrays return the frobenius norm
    if min(X.shape) == 1:
        return np.sqrt((X.reshape(-1) ** 2).sum())

    return svds(X, k=1, which='SM', solver=solver, **kws)[1].item()


def leading_sval(X, solver='lobpcg', **kws):
    """
    Computes the smallest singular value of a matrix using
    scipy.sparse.linalg.svds

    Parameters
    ----------
    X: array-like

    solver: str
        Which solver to use. Must be one of ['lobpcg', 'arpack']

    **kws
        Kws for svds

    Output
    ------
    largest_sval: float
        The largest singular value of X
    """
    # for 1d arrays return the frobenius norm
    if min(X.shape) == 1:
        return np.sqrt((X.reshape(-1) ** 2).sum())

    return svds(X, k=1, which='LM', solver=solver, **kws)[1].item()
