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


def get_diff_mat(d, k=1):
    """
    Gets the kth difference matrix. See Section 2.1.2 of (Tibshirani and Taylor, 2011).

    Parameters
    ----------
    d: int
        Number of points.

    k: int
        Order of the difference.

    Output
    ------
    D: array-like, shape (d - k, d)
        The kth order difference matrix returned in a sparse matrix format.

    References
    ----------
    Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.
    """
    if k == 1:
        D = diags(diagonals=[-np.ones(d), np.ones(d - 1)], offsets=[0, 1])
        D = D.tocsc()
        D = D[:-1, :]
        return D

    else:
        D1d = get_diff_mat(d=d-k+1, k=1)  # first order diff for d - k + 1
        Dkm1 = get_diff_mat(d=d, k=k-1)  # k-1 order diff
        return D1d @ Dkm1
