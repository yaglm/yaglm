import numpy as np

from ya_glm.utils import is_multi_response
from ya_glm.linalg_utils import leading_sval
from ya_glm.processing import process_weights_group_lasso
from ya_glm.opt.huber_regression import huber_grad

def lasso_max(X, y, fit_intercept, loss_func, loss_kws={}, weights=None):

    if is_multi_response(y):

        resp_maxes = []
        for c in range(y.shape[1]):
            if weights is not None:
                w = weights[:, c]
            else:
                w = None

            m = lasso_max(X=X, y=y[:, c],
                          fit_intercept=fit_intercept,
                          loss_func=loss_func,
                          loss_kws=loss_kws,
                          weights=w)

            resp_maxes.append(m)
        return max(resp_maxes)

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        loss_func=loss_func, loss_kws=loss_kws)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        grad = grad[penalized_mask] / weights[penalized_mask]

    return abs(grad).max()


def get_L1toL2_max(X, y, fit_intercept, loss_func, loss_kws={}, weights=None):

    assert loss_func in _MULTI_RESP_LOSSES

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        loss_func=loss_func, loss_kws=loss_kws)

    row_norms = np.linalg.norm(grad, axis=1)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        row_norms = row_norms[penalized_mask] / weights[penalized_mask]

    return max(row_norms)


def group_lasso_max(X, y, groups, fit_intercept, loss_func,
                    loss_kws={}, weights=None):

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        loss_func=loss_func, loss_kws=loss_kws)

    group_norms = np.array([np.linalg.norm(grad[grp_idxs])
                            for grp_idxs in groups])

    weights = process_weights_group_lasso(groups=groups, weights=weights)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        group_norms = group_norms[penalized_mask] / weights[penalized_mask]

    return group_norms.max()


def nuclear_norm_max(X, y, fit_intercept, loss_func,
                     loss_kws={}, weights=None):

    assert loss_func in _MULTI_RESP_LOSSES

    # TODO: double check this is right
    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        loss_func=loss_func, loss_kws=loss_kws)

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


def grad_at_zero(X, y, fit_intercept, loss_func, loss_kws={}):

    if loss_func == 'lin_reg':
        return g0_lin_reg(X=X, y=y, fit_intercept=fit_intercept)

    elif loss_func == 'lin_reg_mr':
        return g0_lin_reg_mr(X=X, y=y, fit_intercept=fit_intercept)

    elif loss_func == 'log_reg':
        return g0_log_reg(X=X, y=y, fit_intercept=fit_intercept)

    elif loss_func == 'huber_reg':
        return g0_log_reg(X=X, y=y, fit_intercept=fit_intercept, **loss_kws)

    elif loss_func == 'huber_reg_mr':
        return g0_log_reg(X=X, y=y, fit_intercept=fit_intercept, **loss_kws)

    else:
        raise NotImplementedError("{} not supported".format(loss_func))


def g0_lin_reg(X, y, fit_intercept):
    if fit_intercept:
        grad = X.T @ (y - np.mean(y))
    else:
        grad = X.T @ y

    return grad / X.shape[0]


def g0_lin_reg_mr(X, y, fit_intercept):
    if fit_intercept:
        grad = X.T @ (y - y.mean(axis=0))
    else:
        grad = X.T @ y

    return grad / X.shape[0]


def g0_huber_reg(X, y, fit_intercept, knot):
    if fit_intercept:
        grad = huber_grad(y - np.mean(y), knot=knot)
    else:
        grad = huber_grad(y, knot=knot)

    return grad / X.shape[0]


def g0_huber_reg_mr(X, y, fit_intercept, knot):
    if fit_intercept:
        grad = huber_grad(y - y.mean(axis=0), knot=knot)
    else:
        grad = huber_grad(y, knot=knot)

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
