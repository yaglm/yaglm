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


def tikhonov_penalty(coef, tikT_tik):
    """
    Parameters
    ----------
    tik_tik: array-like, shape (n_features, n_features)
        The squared tikhonov matrix tikT_tik = tikhonov.T @ tikhonov

    """
    return 0.5 * cp.quad_form(coef, tikT_tik)
