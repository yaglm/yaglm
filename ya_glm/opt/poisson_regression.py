import numpy as np

from ya_glm.opt.utils import safe_data_mat_coef_dot, safe_data_mat_coef_mat_dot
from ya_glm.opt.base import Func


class PoissonRegLoss(Func):
    """
    The Poisson regression loss function

    f(coef, intercept) = (1 / n_samples) sum_{i=1}^n e^{z_i} - y_i z_i
    z_i = x_i.T @ coef + intercept

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    fit_intercept: bool
        Whether or not to include the intercept term.
    """
    def __init__(self, X, y, fit_intercept=True):  # , exposure=None

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y

        # if exposure is not None:
        #     self.log_exposure = np.log(exposure)
        # else:
        #     self.log_exposure = None

    def _eval(self, x):
        z = safe_data_mat_coef_dot(X=self.X, coef=x,
                                   fit_intercept=self.fit_intercept)

        # if self.log_exposure is not None:
        #     z += self.log_exposure

        return (1 / self.X.shape[0]) * (np.exp(z).sum() - z.T @ self.y)

    def _grad(self, x):
        z = safe_data_mat_coef_dot(X=self.X, coef=x,
                                   fit_intercept=self.fit_intercept)

        # if self.log_exposure is not None:
        #     z += self.log_exposure

        r = np.exp(z) - self.y
        coef_grad = (1/self.X.shape[0]) * self.X.T @ r

        if self.fit_intercept:
            intercept_grad = np.mean(r)
            return np.concatenate([[intercept_grad], coef_grad])

        else:
            return coef_grad


class PoissonRegMultiRespLoss(Func):
    """
    The Poisson regression loss function with multiple responses

    f(coef, intercept) = (1 / n_samples) sum_{i=1}^n e^{z_i} - y_i z_i
    z_i = x_i.T @ coef + intercept

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    fit_intercept: bool
        Whether or not to include the intercept term.
    """
    def __init__(self, X, y, fit_intercept=True):  # , exposure=None

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y

        # if exposure is not None:
        #     self.log_exposure = np.log(exposure)
        # else:
        #     self.log_exposure = None

        if self.fit_intercept:
            self.coef_shape = (X.shape[1] + 1, y.shape[1])
        else:
            self.coef_shape = (X.shape[1], y.shape[1])

    def _eval(self, x):
        z = safe_data_mat_coef_mat_dot(X=self.X,
                                       coef=x.reshape(self.coef_shape),
                                       fit_intercept=self.fit_intercept)

        # if self.log_exposure is not None:
        #     z += self.log_exposure

        return (1 / self.X.shape[0]) * (np.exp(z).sum() - (z.T @ self.y).sum())

    def _grad(self, x):
        z = safe_data_mat_coef_mat_dot(X=self.X,
                                       coef=x.reshape(self.coef_shape),
                                       fit_intercept=self.fit_intercept)

        # if self.log_exposure is not None:
        #     z += self.log_exposure

        r = np.exp(z) - self.y
        coef_grad = (1/self.X.shape[0]) * self.X.T @ r

        if self.fit_intercept:
            intercept_grad = r.mean(axis=0)
            return np.vstack([intercept_grad, coef_grad])

        else:
            return coef_grad
