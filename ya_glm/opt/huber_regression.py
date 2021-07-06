import numpy as np

from ya_glm.opt.utils import safe_data_mat_coef_dot
from ya_glm.opt.base import Func
from ya_glm.opt.utils import safe_vectorize
from ya_glm.opt.linear_regression import get_lin_reg_lip


def huber_eval_1d(x, kink=1):
    x_abs = abs(x)
    if x_abs <= kink:
        return 0.5 * x ** 2
    else:
        return kink * (x_abs - 0.5 * kink)


_vec_huber_eval = safe_vectorize(huber_eval_1d)


def huber_eval(x, kink=1):
    return _vec_huber_eval(x, kink).sum()


def huber_grad_1d(x, kink=1):

    if abs(x) <= kink:
        return x
    else:
        return kink * np.sign(x)


_vec_huber_grad = safe_vectorize(huber_grad_1d)


def huber_grad(x, kink=1):
    return _vec_huber_grad(x, kink).sum()


class HuberRegLoss(Func):
    """
    The huber regression loss function

    f(coef, intercept) = 1/ (2 * n_samples) huber(y - X * coef + intercept)

    where huber(z) = TODO

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    kink: float
        The kink point for the huber function.

    fit_intercept: bool
        Whether or not to include the intercept term.

    lip: None, float
        The (optional) precomputed Lipshitz constant of the gradient.

    """
    def __init__(self, X, y, kink=1, fit_intercept=True, lip=None):

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y
        self.kip = kink

        if lip is None:
            # TODO: this is correct right?
            self._grad_lip = get_lin_reg_lip(X=X,
                                             fit_intercept=fit_intercept)
        else:
            self._grad_lip = lip

    def _eval(self, x):
        pred = safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

        return (1/self.X.shape[0]) * huber_eval(pred - self.y, kink=self.kink)

    def _grad(self, x):
        pred = safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

        # TODO: double check!
        resid = pred - self.y
        coef_grad = (1/self.X.shape[0]) * self.X.T @ huber_grad(resid,
                                                                kink=self.kink)

        if self.fit_intercept:
            intercept_grad = np.mean(resid)
            return np.concatenate([[intercept_grad], coef_grad])

        else:
            return coef_grad
