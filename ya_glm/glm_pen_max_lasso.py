import numpy as np

from ya_glm.utils import is_multi_response
from ya_glm.linalg_utils import leading_sval


def lasso_max(X, y, fit_intercept=True, weights=None,
              model_type='lin_reg'):

    if is_multi_response(y):

        resp_maxes = []
        for c in range(y.shape[1]):
            if weights is not None:
                w = weights[:, c]
            else:
                w = None

            m = lasso_max(X=X, y=y[:, c],
                          fit_intercept=fit_intercept,
                          model_type=model_type,
                          weights=w)

            resp_maxes.append(m)
        return max(resp_maxes)

    grad = grad_at_zero(X, y, fit_intercept, model_type)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        grad = grad[penalized_mask] / weights[penalized_mask]

    return abs(grad).max()


def get_L1toL2_max(X, y, fit_intercept=True, weights=None,
                   model_type='lin_reg'):

    grad = grad_at_zero(X, y, fit_intercept, model_type)

    row_norms = np.linalg.norm(grad, axis=1)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        row_norms = row_norms[penalized_mask] / weights[penalized_mask]

    return max(row_norms)


def group_lasso_max(X, y, groups, fit_intercept=True, weights=None,
                    model_type='lin_reg'):

    grad = grad_at_zero(X, y, fit_intercept, model_type)

    group_norms = np.array([np.linalg.norm(grad[grp_idxs])
                            for grp_idxs in groups])

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        group_norms = group_norms[penalized_mask] / weights[penalized_mask]

    return group_norms.max()


def nuclear_norm_max(X, y, fit_intercept=True, weights=None,
                     model_type='lin_reg'):

    # TODO: double check this is right
    grad = grad_at_zero(X, y, fit_intercept, model_type)

    sval_max = leading_sval(grad)

    if weights is None:
        return sval_max
    else:
        # this is correct if the largest sval has the smallest weight
        # it is still correct otherwise, but could be conservative
        # however we expect large svals to have small weights
        penalized_mask = get_is_pen_mask(weights)
        smallest_weight = weights[penalized_mask].min()
        return sval_max / smallest_weight


def grad_at_zero(X, y, fit_intercept=True, model_type='lin_reg'):

    if model_type == 'lin_reg':
        return g0_lin_reg(X, y, fit_intercept)

    elif model_type == 'lin_reg_mr':
        return g0_lin_reg_mr(X, y, fit_intercept)

    elif model_type == 'log_reg':
        return g0_log_reg(X, y, fit_intercept)

    else:
        raise NotImplementedError("{} not supported".format(model_type))


def g0_lin_reg(X, y, fit_intercept=True):
    if fit_intercept:
        grad = X.T @ (y - np.mean(y))
    else:
        grad = X.T @ y

    return grad / X.shape[0]


def g0_lin_reg_mr(X, y, fit_intercept=True):
    if fit_intercept:
        grad = X.T @ (y - y.mean(axis=0))
    else:
        grad = X.T @ y

    return grad / X.shape[0]


def g0_log_reg(X, y, fit_intercept=True):

    if fit_intercept:
        grad = X.T @ (y.mean() - y)
    else:
        grad = X.T @ (0.5 - y)

    return grad / X.shape[0]


def is_nonzero_weight(w):
    """
    Whether or not an entry of a weights vector is non-zero
    """
    if w is not None \
            and not np.isnan(w) \
            and abs(w) > np.finfo(float).eps:
        return True
    else:
        return False


def get_is_pen_mask(weights):
    """
    Find the entries of a weights vector that are penalized

    Parameters
    ----------
    weights: array-like
        The input weights

    Output
    ------
    non_zero_mask: array-like

    """
    weights = np.array(weights)
    return np.array([is_nonzero_weight(w)
                     for w in weights.reshape(-1)]).\
        reshape(weights.shape)
