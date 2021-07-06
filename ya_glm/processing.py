import numpy as np
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from scipy.sparse import issparse, diags


def process_X(X, standardize=False, copy=True,
              check_input=True, accept_sparse=False,
              allow_const_cols=True):
    """
    Processes and possibly standardize the X feature matrix.
    Here standardization means mean centering then scaling by
    the standard deviation for each column.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    standardize: bool
        Whether or not to mean center then scale by the standard deviations.

    copy: bool
        Copy data matrix or standardize in place.

    check_input: bool
        Whether or not we should validate the input.

    accept_sparse : str, bool or list/tuple of str
        See sklearn.utils.validation.check_array.

    allow_const_cols: bool
        Whether or not to allow constant columns.

    Output
    ------
    X, out

    X: array-like, shape (n_samples, n_features)
        The possibly standardized covariate data.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['X_offset']: array-like, shape (n_features, )
            The column means.

        out['X_scale']: array-like, shape (n_features, )
            The column stds. Note if std=0 we set X_scale to 1.
    """

    out = {}

    # input validation and copying
    if check_input:
        X = check_array(X, copy=copy,
                        accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)
    elif copy:
        if issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

    # compute column STDs
    if standardize or not allow_const_cols:
        X_scale = X.std(axis=0)
        const_cols = X_scale <= np.finfo(float).eps

        # for constant columns, put their scale as 1
        X_scale[const_cols] = 1

    # check for constant columns
    if not allow_const_cols:
        if sum(const_cols) > 0:
            raise ValueError("Constant column detected")

    # center by feature means and scale by feature stds
    if standardize:
        X_offset = X.mean(axis=0)

        out['X_scale'] = X_scale
        out['X_offset'] = X_offset

        if issparse(X):
            raise NotImplementedError
            # TODO: add this

        else:
            # mean center then scale
            X -= X_offset
            X = X @ diags(1.0 / X_scale)

    return X, out


def check_Xy(X, y):
    """
    Make sure X and y have the same number of samples and same data type.
    """
    y = np.asarray(y, dtype=X.dtype)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of rows!")

    return X, y
