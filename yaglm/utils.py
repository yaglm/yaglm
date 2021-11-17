import numpy as np
from copy import deepcopy
from numbers import Number
from itertools import islice

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


def fit_if_unfitted(estimator, X, y=None, **fit_params):
    """
    Fits an estimator if it has not yet been fitted. If it has been fit then will just return the input estimator.

    Parameters
    -----------
    estimator: sklearn estimator
        The estimator.

    X, y:
        The input to estimator.fit(X, y, **fit_params)

    **fit_params:
        Keyword args to be  passed to fit

    Output
    ------
    estimator:
        The fitted estimator.
    """
    if not is_fitted(estimator):
        return clone(estimator).fit(X, y, **fit_params)
    return estimator


def delete_fit_attrs(est):
    """
    Removes any fit attribute from an estimator.
    """
    fit_keys = [k for k in est.__dict__.keys() if k.endswith('_')]
    for k in fit_keys:
        del est.__dict__[k]
    return est


def is_str_and_matches(check, value, lower=True):
    """
    Checks if an input is a string and if it matches a given value.

    Parameters
    ----------
    check:
        The object to check.

    value: str
        The value we want to match.

    lower: bool
        Lower case both check and value before matching.
    """

    if not isinstance(check, str):
        return False

    else:
        if lower:
            return check.lower() == value.lower()
        else:
            return check == value


def get_shapes(n_features, n_responses):
    """
    Gets the shapes of the coefficeint and intercept.

    Parameters
    ----------
    n_features: int
        Number of features.

    n_responses: int
        Number of responses

    Output
    ------
    coef_shape, intercept_shape

    coef_shape: tuple
        Shape of the coefficient.

    intercept_shape: tuple
        Shape of the intercept.
    """
    if n_responses == 1:
        coef_shape = (n_features, )
        intercept_shape = ()
    else:
        coef_shape = (n_features, n_responses)
        intercept_shape = (n_responses, )

    return coef_shape, intercept_shape


def get_shapes_from(X, y):
    """
    Gets the shapes of the coefficeint and intercept.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate matrix.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The response data.

    Output
    ------
    coef_shape, intercept_shape

    coef_shape: tuple
        Shape of the coefficient.

    intercept_shape: tuple
        Shape of the intercept.
    """
    n_features = X.shape[1]

    if y.ndim == 1:
        n_respones = 1
    else:
        n_respones = y.shape[1]

    return get_shapes(n_features=n_features, n_responses=n_respones)


def get_from(iterable, idx):
    """
    Selects an index from an iterable that may be a generator without going through the entire generator.
    Acts like list(iterable)[idx]
    """
    try:
        return next(islice(iterable, idx, idx + 1))
    except StopIteration:
        raise ValueError("idx={} was out of range for iterable".format(idx))


def count_support(x, zero_tol='machine'):
    """
    Counts the number of non-zero elements in an array.
    Parameters
    ----------
    x: array-like
        The array whose support we want to count.

    zero_tol: float, str
        Tolerance for declaring a small value equal to zero. If zero_tol=='machine' then we use the machine precision i.e. np.finfo(x.dtype).eps.

    Output
    ------
    n_nonzero: int
        Number of non zero elements of x.
    """

    _x = np.array(x)
    if zero_tol == 'machine':
        zero_tol = np.finfo(_x.dtype).eps

    return (abs(_x) > zero_tol).sum()


def lb_transform_to_indices(lb, y):
    """
    Transforms the dummay variable encoding output by a LabelBinarizer to integers. Safely returns a numpy vector even if the binerizer returns a sparse output.

    Essentially just

    lb.fit_transform(y).argmax(axis=1),

    but with additional numpy converstion.


    Parameters
    ----------
    lb: sklearn.preprocessing.LabelBinarizer
        The fit label binarizer.

    y: array-like
        The target values to transform

    Output
    ------
    y_idxs: array-like of ints
        The class label indices.
    """
    return np.array(lb.fit_transform(y).argmax(axis=1)).reshape(-1)


def enet_params_from_sum(pen_val_1, pen_val_2):
    """
    Computes the elastic net pen_val and mix_val from the two penalty values.

    pen_val_1 = pen_val * mix_val
    pen_val_2 = pen_val * (1 - mix_val )

    Parameters
    ----------
    pen_val_1, pen_val_2: float
        The two penalty values in the sum.

    Output
    ------
    pen_val, mix_val: float
        The elastic net parameters.
    """
    pen_val = pen_val_1 + pen_val_2
    mix_val = pen_val_1 / (pen_val_1 + pen_val_2)
    return pen_val, mix_val
