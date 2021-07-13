import numpy as np
from copy import deepcopy
from inspect import signature


def get_sequence_decr_max(max_val=1, min_val_mult=1e-3, num=20,
                          spacing='lin', decr=True):
    """
    Gets a tuning parameter sequence decreasing from a maximum value.

    Parameters
    ----------
    max_val: float
        The largest value in the sequence.

    min_val_mult: float
        Minimum value = max_val  * min_val_mult

    num: int
        Number of values in the sequence.

    spacing: str
        Determines how points are spaced; must be on of ['lin', 'log'].

    decr: Bool
        Return the sequence in decreasing order.

    Output
    ------
    seq: array-like of floats
        The sequence.

    """
    assert spacing in ['lin', 'log']
    assert min_val_mult <= 1

    min_val = min_val_mult * max_val

    if spacing == 'log':
        assert min_val > 0

    # compute the sequence
    if spacing == 'lin':
        seq = np.linspace(start=min_val,
                          stop=max_val, num=num)

    elif spacing == 'log':
        seq = np.logspace(start=np.log10(min_val),
                          stop=np.log10(max_val),
                          num=num)

    if decr:
        seq = seq[::-1]

    return seq


def get_enet_ratio_seq(num=10, min_val=0.1):
    """
    Returns a sequence values for tuning the l1_ratio parameter of ElasticNet.
    As suggested by sklearn.linear_model.ElasticNetCV, we pick values that
    favor larger values of l1_ratio (meaning more lasso).


    In deatil, the sequence is logarithmicly spaced between 1 and min_val.

    Parameters
    ----------
    num: int
        Number of values to return.

    min_val: float
        The smallest value of l1_ratio to return.

    Output
    ------
    values: array-like, shape (num, )
        The values.
    """
    return 1 + min_val - np.logspace(start=0, stop=np.log10(min_val), num=10)


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

