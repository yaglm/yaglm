import cvxpy as cp


def lin_reg_loss(X, y, coef, intercept=None):
    """
    1/(2 *n_samples) ||X @ ceof + intercept - y||_2^2
    """
    pred = X @ coef
    if intercept is not None:
        pred += intercept

    return (0.5 / X.shape[0]) * cp.sum_squares(pred - y)


def log_reg_loss(X, y, coef, intercept=None):
    pred = X @ coef
    if intercept is not None:
        pred += intercept

    return (1 / X.shape[0]) * cp.sum(cp.logistic(pred) - cp.multiply(y, pred))


def quantile_reg_loss(X, y, coef, intercept=None, quantile=0.5):
    pred = X @ coef
    if intercept is not None:
        pred += intercept

    return (1 / X.shape[0]) * cp.sum(tilted_L1(y - pred, quantile=quantile))


def tilted_L1(u, quantile=0.5):
    """
    tilted_L1(u; quant) = quant * [u]_+ + (1 - quant) * [u]_
    """
    return 0.5 * cp.abs(u) + (quantile - 0.5) * u
