from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.sparse import issparse

import numpy as np


def weighted_mean_std(X, sample_weight=None, ddof=0):
    """
    Computes possible weighted mean and standard deviations of each column of a data matrix. It is safe to call this function on either a sparse or dense matrix.

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features)
        The data matrix.

    sample_weight: None, array-like shape (n_samples)
        The optional sample weights to use.

    ddof: int
        The divisor used in calculations
        is ``TOT_WEIGHT - ddof``, where ``TOT_WEIGHT`` is the total weight.
        If sample_weight is None or norm_weight=True then TOT_WEIGHT = n_samples.
        Otherwise, TOT_WEIGHT = sample_weight.sum()

    Output
    ------
    mean, std

    mean: array-like, shape (n_features, )
        The weighted mean for each feature.

    std: array-like, shape (n_features, )
        The weighted standard deviation for each feature.
    """

    n_samples = X.shape[0]

    # process sample weights
    if sample_weight is not None:
        _sample_weight = np.array(sample_weight).reshape(-1).astype(X.dtype)
        assert len(_sample_weight) == n_samples

        # normalize the weights
        _sample_weight /= _sample_weight.sum()
        _sample_weight *= n_samples

        TOT_WEIGHT = _sample_weight.sum()

    else:
        TOT_WEIGHT = n_samples
        _sample_weight = None

    # sklearn has this built in for sparse matrices
    # TODO: can we find this somewhere for dense?
    if issparse(X):

        # TODO: handle ddof
        MEAN, VAR, SUM_WEIGHTS = \
            mean_variance_axis(X=X, axis=0, weights=_sample_weight,
                               return_sum_weights=True)

        VAR *= SUM_WEIGHTS / (TOT_WEIGHT - ddof)
        return MEAN, np.sqrt(VAR)

    # unweighted, dense case
    if sample_weight is None:
        return X.mean(axis=0), X.std(axis=0, ddof=ddof)

    else:  # weighted, dense case
        MEAN = X.T @ _sample_weight / TOT_WEIGHT
        VAR = ((X - MEAN) ** 2).T @ _sample_weight
        VAR = VAR / (TOT_WEIGHT - ddof)

        return MEAN, np.sqrt(VAR)
