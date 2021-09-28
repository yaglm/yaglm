import cvxpy as cp
import numpy as np


def lin_reg_loss(X, y, coef, intercept=None):
    """
    1/(2 *n_samples) ||X @ ceof + intercept - y||_2^2
    """
    z = get_z(X, coef, intercept)

    return (0.5 / X.shape[0]) * cp.sum_squares(z - y)


def l2_reg_loss(X, y, coef, intercept=None):

    z = get_z(X, coef, intercept)

    return (1 / np.sqrt(X.shape[0])) * cp.norm(z - y, axis=0)


def log_reg_loss(X, y, coef, intercept=None):
    z = get_z(X, coef, intercept)

    return (1 / X.shape[0]) * cp.sum(cp.logistic(z) - cp.multiply(y, z))


def multinomial_loss(X, y, coef, intercept=None):

    z = get_z(X, coef, intercept)
    tops = cp.log_sum_exp(z, axis=1)
    bots = cp.sum(cp.multiply(y, z), axis=1)
    return (1 / X.shape[0]) * cp.sum(bots - tops)


def poisson_reg_loss(X, y, coef, intercept=None):
    z = get_z(X, coef, intercept)

    return (1 / X.shape[0]) * cp.sum(cp.exp(z) - cp.multiply(y, z))


def quantile_reg_loss(X, y, coef, intercept=None, quantile=0.5):
    z = get_z(X, coef, intercept)

    return (1 / X.shape[0]) * cp.sum(tilted_L1(y - z, quantile=quantile))


def tilted_L1(u, quantile=0.5):
    """
    tilted_L1(u; quant) = quant * [u]_+ + (1 - quant) * [u]_
    """
    return 0.5 * cp.abs(u) + (quantile - 0.5) * u


def huber_reg_loss(X, y, coef, intercept=None, knot=1):
    z = get_z(X, coef, intercept)
    # cvxpy's huber is missing the 1/2
    return (0.5 / X.shape[0]) * cp.sum(cp.huber(y - z, M=knot))


# def hinge_loss(X, y, coef, intercept=None):
# TODO: need to make y -1, 1
#     z = get_z(X, coef, intercept)
#     return (1 / X.shape[0]) * cp.sum(cp.pos(1 - cp.multiply(y, z)))


def get_z(X, coef, intercept=None):
    z = X @ coef
    if intercept is not None:
        z += intercept
    return z
