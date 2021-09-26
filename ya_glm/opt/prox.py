import numpy as np
from ya_glm.linalg_utils import euclid_norm


def soft_thresh(vec, thresh_vals):
    """
    The soft thresholding operator.

    Parameters
    ----------
    vec: array-like
        The values to threshold

    thresh_vals: float, array-like
        The thresholding values

    Output
    -------
    vec_thresh: array-like
    """
    return np.sign(vec) * np.fmax(abs(vec) - thresh_vals, 0)

# TODO: is this useful? If not remove it
# def prox_ridge_lasso(x, lasso_pen_val=1, lasso_weights=None,
#                      ridge_pen_val=1, ridge_weights=None, step=1):
#     """
#     Evaluates the proximal operator of

#     f(x; step) = lasso_mul * sum_j lasso_weights_j |x_j|
#         + 0.5 * ridge_mul * sum_j lasso_weights_j x_j^2

#     Parameters
#     ----------
#     x: array-like
#         The value at which to evaluate the prox operator.

#     lasso_pen_val: float
#         The multiplicative penalty value for the lasso penalty.

#     lasso_weights: None, array-like
#         The (optional) variable weights for the lasso penalty.

#     ridge_pen_val: float
#         The multiplicative penalty value for the ridge penalty.

#     ridge_weights: None, array-like
#         The (optional) variable weights for the ridge penalty.

#     step: float
#         The step size.

#     Output
#     ------
#     prox_val: array-like
#         The proximal operator.
#     """

#     lasso_pen_val = lasso_pen_val * step
#     ridge_pen_val = ridge_pen_val * step

#     if lasso_weights is None:
#         lasso_weights = np.ones_like(x)
#     thresh = lasso_pen_val * np.array(lasso_weights)

#     if ridge_weights is None:
#         ridge_weights = np.ones_like(x)
#     mult = ridge_pen_val * np.array(ridge_weights)
#     mult = 1 / (1 + mult)

#     return soft_thresh(x * mult, thresh * mult)

# TODO: is this useful? If not remove it.
# def prox_ridge_perturb(x, prox, ridge_pen_val=1, step=1):
#     """
#     Evaluates the proximal operator of

#     f(x) + 0.5 * ridge_pen_val ||x||_2^2

#     e.g. see Theorem 6.13 of (Beck, 2017).

#     Parameters
#     ----------
#     x: array-like
#         The value at which to evaluate the prox operator.

#     prox: callable(x, step) -> array-like
#         The proximal operator of f.

#     ridge_pen_val: float
#         The ridge penalty value.

#     step: float
#         The step size.

#     Output
#     ------
#     prox_val: array-like
#         The proximal operator.

#     References
#     ----------
#     Beck, A., 2017. First-order methods in optimization. Society for Industrial and Applied Mathematics.
#     """
#     denom = ridge_pen_val * step + 1
#     return prox(x / denom, step=step / denom)


def L2_prox(x, mult):
    """
    Computes the proximal operator of mult * ||x||_2
    """
    norm = euclid_norm(x)

    if norm <= mult:
        return np.zeros_like(x)
    else:
        return x * (1 - (mult / norm))


def squared_l1_prox_pos(x, step=1, weights=None, check=False):
    """
    prox_{step * f}(x) for postive vectors x

    where f(z) = (sum_i w_i |z_i|)^2

    Parameters
    ----------
    x: array-like
        The vector to evaluate the prox at. Note this must be positive.

    step: float
        The prox step size.

    weights: array-like
        The (optional) positive weights.

    check: bool
        Whether or not to check that x is non-negative.

    Output
    ------
    p: array-like
        The value of the proximal operator.

    References
    ----------
    Lin, M., Sun, D., Toh, K.C. and Yuan, Y., 2019. A dual Newton based preconditioned proximal point algorithm for exclusive lasso models. arXiv preprint arXiv:1902.00151.
    """
    if check:
        assert all(x >= 0)

    if weights is None:
        weights = np.ones_like(x)

    # get indices to sort x / weights in decreasing order
    x_over_w = x / weights
    decr_sort_idxs = np.argsort(x_over_w)[::-1]

    x_sort = x[decr_sort_idxs]
    weights_sort = weights[decr_sort_idxs]

    # compute threshold value
    s = np.cumsum(x_sort * weights_sort)
    L = np.cumsum(weights_sort ** 2)
    alpha_bar = max(s / (1 + 2 * step * L))
    thresh = 2 * step * alpha_bar * weights

    # return soft thresholding
    return np.maximum(x - thresh, 0)
