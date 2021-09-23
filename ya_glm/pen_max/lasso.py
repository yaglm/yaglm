import numpy as np

from ya_glm.linalg_utils import leading_sval
from ya_glm.opt.glm_loss.from_config import get_glm_loss_func


def get_lasso_pen_max(X, y, loss, fit_intercept, weights=None,
                      sample_weight=None,
                      multi_task=False, groups=None, nuc=False):

    # make sure only one special thing is provided
    assert sum([multi_task, groups is not None, nuc]) <= 1

    # compute the gradient of the loss function when
    #  the coefficeint is zero
    loss_func = get_glm_loss_func(config=loss, X=X, y=y,
                                  fit_intercept=fit_intercept,
                                  sample_weight=sample_weight)

    grad = loss_func.grad_at_coef_eq0()

    if multi_task:
        return mult_task_lasso_max(grad, weights=weights)

    elif nuc:
        return nuclear_norm_max(grad, weights=weights)

    elif groups is not None:
        return group_lasso_max(grad, groups=groups, weights=weights)

    else:
        return lasso_max(grad, weights=weights)


def lasso_max(grad, weights=None):

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        grad = grad[penalized_mask] / weights[penalized_mask]

    return abs(grad.ravel()).max()


def mult_task_lasso_max(grad, weights=None):

    row_norms = np.linalg.norm(grad, axis=1)

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        row_norms = row_norms[penalized_mask] / weights[penalized_mask]

    return max(row_norms)


def group_lasso_max(grad, groups, weights=None):

    group_norms = np.array([np.linalg.norm(grad[grp_idxs])
                            for grp_idxs in groups])

    if weights is not None:
        penalized_mask = get_is_pen_mask(weights)

        # technically this is a hack but this gives the correct formula
        group_norms = group_norms[penalized_mask] / weights[penalized_mask]

    return group_norms.max()


def nuclear_norm_max(grad, weights=None):

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
