from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone


def is_fitted(estimator):
    """
    Checks if an estimator has been fitted.

    Parameters
    ----------
    estimator: sklearn estimator
        The estimator to check.

    Output
    ------
    bool:
        Returns True iff the estimator has been fitted.
    """

    try:
        check_is_fitted(estimator)
        is_fitted = True
    except NotFittedError:
        is_fitted = False
    return is_fitted


def fit_if_unfitted(estimator, X, y=None):
    """
    Fits an estimator if it has not yet been fitted. If it has been fit then will just return the input estimator.

    Parameters
    -----------
    estimator: sklearn estimator
        The estimator.

    X, y:
        The input to estimator.fit(X, y)

    Output
    ------
    estimator:
        The fitted estimator.
    """
    if not is_fitted(estimator):
        return clone(estimator).fit(X, y)
    return estimator
