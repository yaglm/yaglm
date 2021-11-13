import numpy as np

from yaglm.linalg_utils import leading_sval
from yaglm.sparse_utils import safe_hstack


def safe_covar_mat_op_norm(X, fit_intercept=True):
    """
    Computes the operator norm of  X or [X; 1_n] safely. This works when X is dense, sparse, or a linear operator.

    Parameters
    ---------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    fit_intercept: bool
        Whether or not to include the intercept term.

    Output
    ------
    op_norm: float
    """

    if fit_intercept:
        constant_col = np.ones((X.shape[0], 1))
        X_ = safe_hstack([constant_col, X])
    else:
        X_ = X

    return leading_sval(X_)
