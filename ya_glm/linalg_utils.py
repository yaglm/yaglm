from scipy.sparse.linalg import svds
from scipy.sparse import diags
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


# TODO: add kth difference
def get_diff_mat(d):
    """
    Gets the kth difference matrix; see (2) and(3) from https://arxiv.org/pdf/1406.2082.pdf

    Parameters
    ----------
    d: int
        Number of points.

    k: int
        Order of the difference.

    Output
    ------
    D: array-like, shape (d - k, d)
        The kth order difference matrix.
    """
    D = diags(diagonals=[-np.ones(d), np.ones(d - 1)], offsets=[0, 1])
    D = D.tocsc()
    D = D[:-1, :]
    return D
