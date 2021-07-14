import cvxpy as cp


def lasso_penalty(coef, weights=None):
    if weights is not None:
        return cp.norm1(cp.multiply(coef, weights))
    else:
        return cp.norm1(coef)


def ridge_penalty(coef, weights=None):
    if weights is not None:
        return 0.5 * sum(cp.multiply(coef ** 2, weights))
    else:
        return 0.5 * cp.sum_squares(coef)


def tikhonov_penalty(coef, tikhonov):
    return 0.5 * cp.quad_form(coef, tikhonov.T @ tikhonov)


def multi_task_lasso_penalty(coef, weights=None):

    row_norms = cp.norm(coef, p='fro', axis=1)

    if weights:
        return weights.T @ row_norms
    else:
        return cp.sum(row_norms)


def group_lasso_penalty(coef, groups, weights=None):

    group_norms = [cp.norm(coef[grp_idxs], p='fro') for grp_idxs in groups]

    if weights is None:
        return cp.sum(group_norms)
    else:
        return weights.T @ cp.hstack(group_norms)
