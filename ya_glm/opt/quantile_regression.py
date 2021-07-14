import numpy as np

from ya_glm.opt.base import Func
from ya_glm.opt.utils import safe_vectorize
from ya_glm.opt.utils import safe_data_mat_coef_dot


def tilted_L1(u, quantile=0.5):
    """
    tilted_L1(u; quant) = quant * [u]_+ + (1 - quant) * [u]_
    """
    return 0.5 * abs(u) + (quantile - 0.5) * u


def tilted_L1_prox_1d(x, step, quantile=0.5):
    """
    prox(x) = argmin_z rho_quantile(z) + (0.5 / step) * ||x - z||_2^2

    See Lemma 1 of ADMM for High-Dimensional Sparse Penalized
Quantile Regression
    """
    if step < np.finfo(float).eps:
        return 0

    t_a = quantile * step  # tau / alpha

    if x > t_a:
        return x - t_a

    t_m1_a = (quantile - 1) * step
    if t_m1_a <= x:
        return 0

    else:
        return x - t_m1_a


tilted_L1_prox = safe_vectorize(tilted_L1_prox_1d)


class QuantileRegLoss(Func):
    """
    The quantile regression loss function

    f(coef, intercept) = (1 / n_samples) * sum_{i=1}^n rho(y_i - z_i; quantle)
    where z_i = x_i.T @ coef + intercept

    and rho(r; quantile) is the tilted_L1 function

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    quantile: float
        The quantile.

    fit_intercept: bool
        Whether or not to include the intercept term.

    """
    def __init__(self, X, y,
                 quantile=0.5,
                 fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y
        self.quantile = quantile

    def _eval(self, x):
        pred = safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

        losses = tilted_L1(self.y - pred, quantile=self.quantile)
        return np.mean(losses)
