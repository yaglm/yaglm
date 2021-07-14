import cvxpy as cp


def lasso(coef, weights=None):
    if weights is not None:
        return cp.norm1(cp.multiply(coef, weights))
    else:
        return cp.norm1(coef)


def ridge(coef, weights=None):
    if weights is not None:
        return 0.5 * sum(cp.multiply(coef ** 2, weights))
    else:
        return 0.5 * cp.sum_squares(coef)
