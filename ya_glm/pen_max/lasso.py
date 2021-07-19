import numpy as np

from ya_glm.linalg_utils import leading_sval
from ya_glm.opt.glm_loss.get import get_glm_loss

def get_pen_max(pen_kind, **kws):
    if pen_kind == 'entrywise':
        return lasso_max(**kws)

    elif pen_kind == 'multi_task':
        return get_L1toL2_max(**kws)

    elif pen_kind == 'group':
        return group_lasso_max(**kws)

    elif pen_kind == 'nuc':
        return nuclear_norm_max(**kws)

    else:
        raise ValueError("Bad input for pen_kind: {}".format(pen_kind))


def lasso_max(X, y, fit_intercept, loss_func,
              loss_kws={}, weights=None, sample_weight=None):

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        sample_weight=sample_weight,
                        loss_func=loss_func, loss_kws=loss_kws)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        grad = grad[penalized_mask] / weights[penalized_mask]

    return abs(grad.ravel()).max()


def get_L1toL2_max(X, y, fit_intercept, loss_func, loss_kws={},
                   weights=None, sample_weight=None):

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        sample_weight=sample_weight,
                        loss_func=loss_func, loss_kws=loss_kws)

    row_norms = np.linalg.norm(grad, axis=1)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        row_norms = row_norms[penalized_mask] / weights[penalized_mask]

    return max(row_norms)


def group_lasso_max(X, y, groups, fit_intercept, loss_func,
                    loss_kws={}, weights=None, sample_weight=None):

    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        sample_weight=sample_weight,
                        loss_func=loss_func, loss_kws=loss_kws)

    group_norms = np.array([np.linalg.norm(grad[grp_idxs])
                            for grp_idxs in groups])

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        group_norms = group_norms[penalized_mask] / weights[penalized_mask]

    return group_norms.max()


def nuclear_norm_max(X, y, fit_intercept, loss_func,
                     loss_kws={}, weights=None, sample_weight=None):

    # TODO: double check this is right
    grad = grad_at_zero(X=X, y=y, fit_intercept=fit_intercept,
                        sample_weight=sample_weight,
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


def grad_at_zero(X, y, fit_intercept, loss_func, loss_kws={},
                 sample_weight=None):

    func = get_glm_loss(X=X, y=y,
                        loss_func=loss_func, loss_kws=loss_kws,
                        fit_intercept=fit_intercept,
                        sample_weight=sample_weight)

    return func.grad_at_coef_eq0()


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
