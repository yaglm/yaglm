import numpy as np
from copy import deepcopy
from numbers import Number

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone


def is_multi_response(y):
    """
    Checks whether or not y is a multiple output response.

    Parameters
    ----------
    y: array-like

    Output
    ------
    is_mat: bool
        Whether or not y is a multi-output response
    """

    y = np.array(y)
    if y.ndim == 2 and y.shape[1] >= 2:
        return True
    else:
        return False


def lasso_and_ridge_from_enet(pen_val, l1_ratio):
    """
    Returns the lasso and L2 penalties from the elastic net parameterization

    Parameters
    ----------
    pen_val: float

    l1_ratio: float, None

    Output
    ------
    lasso_pen, ridge_pen
    """

    if l1_ratio is None or l1_ratio == 0:
        lasso_pen = None
        ridge_pen = pen_val

    elif l1_ratio == 1:
        lasso_pen = pen_val
        ridge_pen = None

    else:
        lasso_pen = pen_val * l1_ratio
        ridge_pen = pen_val * (1 - l1_ratio)

    return lasso_pen, ridge_pen


def get_coef_and_intercept(est, copy=False, error=False):
    """
    Extracts the coefficient and intercept from an estimatator.

    Parameters
    ----------
    est:
        The estimator.

    copy: bool
        Whether or not to copy the underlying data.

    error: bool
        If no coefficient is found, whether or not to throw an exception.

    Output
    ------
    coef, intercept

    coef: array-like
        The coefficient.

    intercept: float, array-like, None
        The intercept. Returns None if not found.
    """

    if hasattr(est, 'coef_'):
        coef = est.coef_
        if hasattr(est, 'intercept_'):
            intercept = est.intercept_
        else:
            intercept = None

    elif hasattr(est, 'best_estimator_'):
        # try getting the data from the selected estimator
        return get_coef_and_intercept(est=est.best_estimator_,
                                      copy=copy,
                                      error=error)

    else:
        if error:
            raise ValueError("Unable to detect coefficient")

    if copy:
        return deepcopy(coef), deepcopy(intercept)

    else:
        return coef, intercept


def maybe_add(d, exclude_false=False, **kws):
    """
    Adds keywork argumnts to a dict if their values are not None.

    Parameters
    ----------
    d: dict
        The dictionary to add to.

    exclude_false: bool
        Exclue keys whose values are false.

    kws: dict
        The keys to maybe add

    """
    for k, v in kws.items():
        if v is not None:
            d[k] = v

    return d


def clip_zero(x, zero_tol=1e-8):
    """
    Sets x or the elements of x to zero when they are very small.

    Parameters
    ----------
    x: Number, array-like
        The value or values to clip.

    zero_tol: float
        The tolerance below which we declare a number to be zero.

    Output
    ------
    x_clipped: Number or np.array

    """
    if isinstance(x, Number):
        if abs(x) <= zero_tol:
            return 0
        else:
            return x

    x_ = np.zeros_like(x)
    non_zero_mask = abs(np.array(x)) > zero_tol
    x_[non_zero_mask] = x[non_zero_mask]
    return x_


def at_most_one_none(*args):
    """
    Returns True if at most one of the args is not None.
    """
    n_none = sum([a is None for a in args])
    return n_none <= 1


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
