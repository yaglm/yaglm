import numpy as np

from ya_glm.opt.utils import safe_data_mat_coef_dot, safe_data_mat_coef_mat_dot

from ya_glm.opt.base import Func


class LinRegLoss(Func):
    """
    The linear regression loss function

    f(coef, intercept) = 1/ (2 * n_samples) ||y - X * coef + intercept||_2^2

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    fit_intercept: bool
        Whether or not to include the intercept term.

    lip: None, float
        The (optional) precomputed Lipshitz constant of the gradient.

    """
    def __init__(self, X, y, fit_intercept=True, lip=None):

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y

        if lip is None:
            self._grad_lip = get_lin_reg_lip(X=X, fit_intercept=fit_intercept)
        else:
            self._grad_lip = lip

    def _eval(self, x):
        pred = safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

        return (0.5 / self.X.shape[0]) * ((pred - self.y) ** 2).sum()

    def _grad(self, x):
        pred = safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

        resid = pred - self.y
        coef_grad = (1/self.X.shape[0]) * self.X.T @ resid

        if self.fit_intercept:
            intercept_grad = np.mean(resid)
            return np.concatenate([[intercept_grad], coef_grad])

        else:
            return coef_grad


class LinRegMultiRespLoss(Func):
    """
    The linear regression loss function

    f(coef, intercept) = 1/ (2 * n_samples) ||Y - X * coef + intercept||_2^2

    coef (d x K)
    intercept: K X1

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    Y: array-like, shape (n_samples, n_responses)
        The outcomes.

    fit_intercept: bool
        Whether or not to include the intercept term.

    lip: None, float
        The (optional) precomputed Lipshitz constant of the gradient.

    """
    def __init__(self, X, y, fit_intercept=True, lip=None):

        self.fit_intercept = fit_intercept
        self.X = X
        self.y = y

        if self.fit_intercept:
            self.coef_shape = (X.shape[1] + 1, y.shape[1])
        else:
            self.coef_shape = (X.shape[1], y.shape[1])

        if lip is None:
            self._grad_lip = get_lin_reg_lip(X=X, fit_intercept=fit_intercept)
        else:
            self._grad_lip = lip

    def _eval(self, x):
        pred = safe_data_mat_coef_mat_dot(X=self.X,
                                          coef=x.reshape(self.coef_shape),
                                          fit_intercept=self.fit_intercept)

        return (0.5 / self.X.shape[0]) * ((pred - self.y) ** 2).sum()

    def _grad(self, x):
        pred = safe_data_mat_coef_mat_dot(X=self.X,
                                          coef=x.reshape(self.coef_shape),
                                          fit_intercept=self.fit_intercept)

        resid = pred - self.y
        coef_grad = (1/self.X.shape[0]) * self.X.T @ resid

        if self.fit_intercept:
            intercept_grad = resid.mean(axis=0)
            grad = np.vstack([intercept_grad, coef_grad])

        else:
            grad = coef_grad

        return grad


def get_lin_reg_lip(X, fit_intercept=True):
    """
    Gets the lipschitz constant for the least squares loss function

    f(beta) = 1/(2 * n_samples) ||y - X beta||_2^2

    which is (1/n_samples) * ||X||_op^2

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    fit_intercept: bool
        Whether or not we will fit an intercept. If we do, then we append a column of zeros onto X

    Output
    ------
    lip: float
        The Lipschitz constant

    """
    # TODO: perhaps add option for sparse X
    assert X.ndim == 2

    n_samples = X.shape[0]

    if fit_intercept:
        op_norm = np.linalg.norm(np.hstack([np.ones((n_samples, 1)), X]), ord=2)
    else:
        op_norm = np.linalg.norm(X, ord=2)

    return (1/n_samples) * op_norm ** 2
