import numpy as np

from ya_glm.opt.base import Func
from ya_glm.opt.utils import decat_coef_inter_vec


class WithIntercept(Func):
    def __init__(self, func):
        self.func = func

    def _eval(self, x):
        coef, _ = decat_coef_inter_vec(x)
        return self.func.eval(coef)

    def _grad(self, x):
        coef, _ = decat_coef_inter_vec(x)
        g = self.func.grad(coef)
        return np.concatenate([[0], g])

    def _prox(self, x, step):
        coef, intercept = decat_coef_inter_vec(x)
        p = self.func.prox(coef, step)
        return np.concatenate([[intercept], p])  # TODO: check

    @property
    def grad_lip(self):
        return self.func.grad_lip


class LassoPenalty(Func):
    """
    f(x) = mult * sum_{j=1}^d weights_j |x_j|

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, mult=1.0, weights=None):

        self.mult = mult
        if weights is not None:
            weights = np.array(weights).ravel()
        self.weights = weights

    def _eval(self, x):

        if self.weights is None:
            norm_val = abs(x).sum()

        else:
            norm_val = self.weights.ravel().T @ abs(x)

        return norm_val * self.mult

    def _prox(self, x, step):

        # set thresholding values
        if self.weights is None:
            thresh_vals = step * self.mult
        else:
            thresh_vals = (step * self.mult) * np.array(self.weights)

        # apply soft thresholding
        return soft_thresh(x, thresh_vals)


class RidgePenalty(Func):
    """
    f(x) = 0.5 * mult * sum_{j=1}^d weights_j x_j^2

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, mult=1.0, weights=None):

        self.mult = mult
        if weights is not None:
            weights = np.array(weights).ravel()

        self.weights = weights

        if self.weights is None:
            self._grad_lip = mult
        else:
            self._grad_lip = mult * np.array(self.weights).max()

    def _eval(self, x):

        if self.weights is None:
            norm_val = (x ** 2).sum()

        else:
            norm_val = self.weights.T @ (x ** 2)

        return 0.5 * self.mult * norm_val

    def _prox(self, x, step):

        # set shrinkage values
        if self.weights is None:
            shirnk_vals = step * self.mult
        else:
            shirnk_vals = (step * self.mult) * self.weights

        return x / (1 + shirnk_vals)

    def _grad(self, x):
        coef_grad = x
        if self.weights is not None:
            coef_grad = coef_grad * self.weights
        return self.mult * coef_grad


class TikhonovPenalty(Func):
    """
    f(x) = 0.5 * mult * ||mat x ||_2^2
    # TODO: check this

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, mult=1.0, mat=None):

        self.mult = mult
        self.mat = mat

        if mat is None:
            self._grad_lip = mult
        else:
            # TODO: double check
            self._grad_lip = mult * np.linalg.norm(mat, ord=2) ** 2

            # cache this for gradient computations
            # TODO: get this to work with sparse matrices
            # TODO: prehaps allow precomputed mat_T_mat
            self.mat_T_mat = self.mat.T @ self.mat

    def _eval(self, x):

        if self.mat is None:
            norm_val = (x ** 2).sum()

        else:
            norm_val = ((self.mat @ x) ** 2).sum()

        return 0.5 * self.mult * norm_val

    # def _prox(self, x, step):
    # TODO: think about this

    def _grad(self, x):
        grad = x
        if self.mat is not None:
            grad = self.mat_T_mat @ grad

        return self.mult * grad


class LassoRidgePenalty(Func):
    """
    f(x) = lasso_mul * sum_j lasso_weights_j |x_j|
        + 0.5 * ridge_mul * sum_j lasso_weights_j x_j^2


    Parameters
    ----------
    lasso_mult: float
        The multiplicative penalty value for the lasso penalty.

    lasso_weights: None, array-like
        The (optional) variable weights for the lasso penalty.

    ridge_mult: float
        The multiplicative penalty value for the ridge penalty.

    ridge_weights: None, array-like
        The (optional) variable weights for the ridge penalty.

    """
    def __init__(self, lasso_mult=1.0, lasso_weights=None,
                 ridge_mult=1.0, ridge_weights=None):

        self.lasso = LassoPenalty(mult=lasso_mult,
                                  weights=lasso_weights)

        self.ridge = RidgePenalty(mult=ridge_mult,
                                  weights=ridge_weights)

    def _eval(self, x):
        return self.lasso.eval(x) + self.ridge.eval(x)

    def _prox(self, x, step):

        return prox_ridge_lasso(x=x,
                                lasso_mult=step * self.lasso.mult,
                                lasso_weights=self.lasso.weights,
                                ridge_mult=step * self.ridge.mult,
                                ridge_weights=self.ridge.weights)


def soft_thresh(vec, thresh_vals):
    """
    The soft thresholding operator.

    Parameters
    ----------
    vec: array-like
        The values to threshold

    thresh_vals: float, array-like
        The thresholding values

    Output
    -------
    vec_thresh: array-like
    """
    return np.sign(vec) * np.fmax(abs(vec) - thresh_vals, 0)


def prox_ridge_lasso(x, lasso_mult=1,
                     lasso_weights=None, ridge_mult=1, ridge_weights=None):
    """
    Evaluates the proximal operator of

    f(x) = lasso_mul * sum_j lasso_weights_j |x_j|
        + 0.5 * ridge_mul * sum_j lasso_weights_j x_j^2

    Parameters
    ----------
    x: array-like
        The value at which to evaluate the prox operator.

    lasso_mult: float
        The multiplicative penalty value for the lasso penalty.

    lasso_weights: None, array-like
        The (optional) variable weights for the lasso penalty.

    ridge_mult: float
        The multiplicative penalty value for the ridge penalty.

    ridge_weights: None, array-like
        The (optional) variable weights for the ridge penalty.

    Output
    ------
    prox: array-like
        The proximal operator.
    """

    if lasso_weights is None:
        lasso_weights = np.ones_like(x)
    thresh = lasso_mult * np.array(lasso_weights)

    if ridge_weights is None:
        ridge_weights = np.ones_like(x)
    mult = ridge_mult * np.array(ridge_weights)
    mult = 1 / (1 + mult)

    return soft_thresh(x * mult, thresh * mult)
