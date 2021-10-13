import numpy as np

from yaglm.linalg_utils import leading_sval
from yaglm.sparse_utils import safe_hstack, safe_row_scaled


def safe_covar_mat_op_norm(X, fit_intercept=True, sample_weight=None):
    """
    Computes the operator norm of diag(sqrt(w)) X or diag(sqrt(w)) [X; 1_n] safely. This works when X is dense, sparse, or a linear operator.

    Parameters
    ---------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    fit_intercept: bool
        Whether or not to include the intercept term.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    Output
    ------
    op_norm: float
    """

    if fit_intercept:
        constant_col = np.ones((X.shape[0], 1))
        X_ = safe_hstack([constant_col, X])
    else:
        X_ = X

    if sample_weight is not None:
        X_ = safe_row_scaled(mat=X_, s=1 / np.sqrt(sample_weight))

    return leading_sval(X_)
