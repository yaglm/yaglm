import cvxpy as cp


def zero(coef, weights=None):
    return 0


def lasso_penalty(coef, pen_val, weights=None):
    if weights is not None:
        return pen_val * cp.norm1(cp.multiply(coef, weights))
    else:
        return pen_val * cp.norm1(coef)


def ridge_penalty(coef, pen_val, weights=None):
    if weights is not None:
        return pen_val * 0.5 * sum(cp.multiply(coef ** 2, weights))
    else:
        return pen_val * 0.5 * cp.sum_squares(coef)


# def elastic_net_penalty(coef, pen_val, mix_val, weights=None):

#     weights = {} if weights is None else weights
#     return lasso_penalty(coef,
#                          pen_val=pen_val * mix_val,
#                          weights=weights.get('lasso', None)) + \
#         ridge_penalty(coef,
#                       pen_val=pen_val * (1 - mix_val),
#                       weights=weights.get('ridge', None))


def gen_ridge_penalty(coef, pen_val, mat, weights=None):
    return ridge_penalty(mat @ coef, pen_val=pen_val, weights=weights)


def multi_task_lasso_penalty(coef, pen_val, weights=None):

    row_norms = cp.norm(coef, axis=1)

    if weights is not None:
        return pen_val * weights.T @ row_norms
    else:
        return pen_val * cp.sum(row_norms)


def group_lasso_penalty(coef, pen_val, groups, weights=None):

    group_norms = [cp.norm(coef[grp_idxs]) for grp_idxs in groups]

    if weights is not None:
        return pen_val * weights.T @ cp.hstack(group_norms)
    else:
        return pen_val * cp.sum(group_norms)


def gen_lasso_penalty(coef, pen_val, mat, weights=None):
    return lasso_penalty(mat @ coef, pen_val=pen_val, weights=weights)
