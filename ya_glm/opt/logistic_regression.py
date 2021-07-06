import numpy as np

from ya_glm.opt.utils import safe_data_mat_coef_dot
from ya_glm.opt.base import Func


class LogRegLoss(Func):
    """
    The logistic regression loss function

    f(coef, intercept) =
        (1/n_samples) sum_{i=1}^n y_i log(sigma(x_i^T coef + intercept))
        + (1 - y_i) log(sigma(x_i^T coef + intercept))

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    y: array-like, shape (n_samples, )
        The outcomes.

    fit_intercept: bool
        Whether or not to include the intercept term.

    lip: None, float
        The (optional) precomputed Lipshitz constant.

    """
    def __init__(self, X, y, fit_intercept=True, lip=None):

        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept

        if lip is None:
            self._grad_lip = get_log_reg_lip(X=X, fit_intercept=fit_intercept)
        else:
            self._grad_lip = lip

    def _eval(self, x):
        return logistic_loss(X=self.X, y=self.y,
                             coef=x, fit_intercept=self.fit_intercept)

    def _grad(self, x):
        return logistic_loss_grad(X=self.X, y=self.y,
                                  coef=x, fit_intercept=self.fit_intercept)


def get_log_reg_lip(X, fit_intercept=True):
    """
    Gets the lipschitz constant for the logistic regression loss function

    f(coef) = (1/n_samples) sum_{i=1}^n y_i log(sigma(x_i^T coef))
        + (1 - y_i) log(sigma(x_i^T coef))

    where sigma() is the sigmoid function.

    This value is (0.25 / n_samples) * ||X||_op^2

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

    return (0.25/n_samples) * op_norm ** 2


def logistic_loss(X, y, coef, fit_intercept=True):
    """Logistic loss, numerically stable implementation.
    See http://fa.bianp.net/blog/2019/evaluate_logistic/

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Data matrix

    y: array-like, shape (n_samples,)
        The binary lables.

    coef: array-like, shape (n_features,) or  (n_features + 1,)
        Coefficients. If there is an intercept it sholuld be in the first coordinate.

    fit_intercept: bool
        Whether or not the first coordinate of coef is the intercept.

    Returns
    -------
    loss: float
    """
    z = safe_data_mat_coef_dot(X, coef, fit_intercept)
    return np.mean((1 - y) * z - logsig(z))


def logistic_loss_grad(X, y, coef, fit_intercept=True):
    """Computes the gradient of the logistic loss. See http://fa.bianp.net/blog/2019/evaluate_logistic/.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Data matrix

    y: array-like, shape (n_samples,)
        The binary lables.

    coef: array-like, shape (n_features,) or  (n_features + 1,)
        Coefficients. If there is an intercept it sholuld be in the first coordinate.

    fit_intercept: bool
        Whether or not the first coordinate of coef is the intercept.

    Returns
    -------
    grad: array-like, shape (n_features,)
    """
    z = safe_data_mat_coef_dot(X, coef, fit_intercept)
    # s = expit_b(z, y)
    # return X.T.dot(s) / X.shape[0]

    z0_b = expit_b(z, y)

    grad = X.T.dot(z0_b) / X.shape[0]
    grad = np.asarray(grad).ravel()

    if fit_intercept:
        grad_intercept = z0_b.mean()
        grad = np.concatenate(([grad_intercept], grad))

    return grad


def logsig(x):
    """
    Compute the log-sigmoid function component-wise, see http://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def expit_b(x, b):
    """
    Compute sigmoid(x) - b component-wise, see http://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out
