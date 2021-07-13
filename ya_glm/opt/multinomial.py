import numpy as np
from scipy.special import logsumexp

from ya_glm.opt.base import Func
from ya_glm.opt.utils import safe_data_mat_coef_mat_dot, safe_entrywise_mult

# TODO: possibly adjust for numerical stability
# e.g. see https://chrisyeh96.github.io/2018/06/11/logistic-regression.html

# TODO: think carefully implementation speed + stability


class MultinomialLoss(Func):
    """
    Multinomial logistic regression loss function

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data matrix.

    Y: array-like, shape (n_samples, n_classes)
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
            self.opt_var_shape = (X.shape[1] + 1, self.y.shape[1])
        else:
            self.opt_var_shape = (X.shape[1], self.y.shape[1])

        if lip is None:
            self._grad_lip = get_multinomial_lip(X=X,
                                                 fit_intercept=fit_intercept)
        else:
            self._grad_lip = lip

    def _eval(self, x):
        z = safe_data_mat_coef_mat_dot(X=self.X,
                                       coef=x.reshape(self.opt_var_shape),
                                       fit_intercept=self.fit_intercept)

        log_probs = z - logsumexp(z, axis=1)[:, np.newaxis]
        L = - safe_entrywise_mult(self.y, log_probs).sum()

        return L / self.X.shape[0]

    def _grad(self, x):
        z = safe_data_mat_coef_mat_dot(X=self.X,
                                       coef=x.reshape(self.opt_var_shape),
                                       fit_intercept=self.fit_intercept)

        log_probs = z - logsumexp(z, axis=1)[:, np.newaxis]
        probs = np.exp(log_probs)
        diff = probs - self.y
        diff = np.array(diff)  # when y is sparse diff is annoyingly a np.matrix

        coef_grad = (1 / self.X.shape[0]) * self.X.T @ diff

        if self.fit_intercept:
            intercept_grad = diff.mean(axis=0)
            grad = np.vstack([intercept_grad, coef_grad])

        else:
            grad = coef_grad

        return grad


def get_multinomial_lip(X, fit_intercept=True):
    # TODO: I think this is right but double check
    assert X.ndim == 2

    n_samples = X.shape[0]

    if fit_intercept:
        op_norm = np.linalg.norm(np.hstack([np.ones((n_samples, 1)), X]), ord=2)
    else:
        op_norm = np.linalg.norm(X, ord=2)

    return (1 /n_samples) * op_norm ** 2
