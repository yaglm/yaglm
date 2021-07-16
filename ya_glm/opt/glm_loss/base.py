from ya_glm.opt.base import Func
from ya_glm.opt.utils import safe_data_mat_coef_dot, safe_data_mat_coef_mat_dot
from scipy.sparse import diags
import numpy as np

from ya_glm.autoassign import autoassign

# TODO how to handle loss kws
# currently pass them in as a dict but this seems wrong e.g.
# they should be arguments to init


class Glm(Func):
    """
    Represents a GLM loss function.
    (1/n) sum_{i=1}^n w_i L(x_i^T coef + intercept, y_i)
    """
    compute_lip = None

    @autoassign
    def __init__(self, X, y, loss_kws={},
                 fit_intercept=True, sample_weight=None):

        self._set_shape_data()
        self._set_lip_data()

    def _set_shape_data(self):
        # set coefficient shapes
        if self.y.ndim in [0, 1]:
            self.coef_shape_ = (self.X.shape[1], )
            self.intercept_shape_ = (1, )
            if self.fit_intercept:
                self.var_shape_ = (self.X.shape[1] + 1, )
            else:
                self.var_shape_ = self.coef_shape_

        else:
            self.coef_shape_ = (self.X.shape[1], self.y.shape[1])
            self.intercept_shape_ = (self.y.shape[1], )

            if self.fit_intercept:
                self.var_shape_ = (self.X.shape[1] + 1, self.y.shape[1])
            else:
                self.var_shape_ = self.coef_shape_

    def _set_lip_data(self):

        if self.compute_lip is not None:
            self._grad_lip = \
                self.compute_lip(X=self.X,
                                 fit_intercept=self.fit_intercept,
                                 sample_weight=self.sample_weight,
                                 **self.loss_kws)

    def get_z(self, x):
        return safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

    def cat_intercept_coef(self, intercept, coef):
        return np.concatenate([[intercept], coef])

    def get_zero_coef(self):
        return np.zeros(self.coef_shape_)

    def sample_losses(self, z):
        """
        Parameters
        ----------
        z: array-like, (n_samples, )

        Output
        ------
        loss_vals: array-like, (n_samples, )
        """
        raise NotImplementedError

    def sample_grads(self, z):
        """
        Computes the gradient of the prediction at each sample

        Parameters
        ----------
        z: array-like, (n_samples, )

        Output
        ------
        loss_grads: array-like, (n_samples, ) or (n_samples, n_responses)
        """
        raise NotImplementedError

    def _eval(self, x):
        z = self.get_z(x)
        losses = self.sample_losses(z=z, y=self.y, **self.loss_kws)
        if self.sample_weight is not None:
            losses *= self.sample_weight
        return losses.sum() / self.X.shape[0]

    def _grad(self, x):
        # compute loss at each sample
        z = self.get_z(x)
        loss_grads = self.sample_grads(z=z, y=self.y, **self.loss_kws)

        # possibly reweight
        if self.sample_weight is not None:
            loss_grads = diags(self.sample_weight) @ loss_grads

        # get coefficient gradients
        grad = self.X.T @ loss_grads

        # possibly add intercept to gradient
        if self.fit_intercept:
            intercept_grad = loss_grads.sum(axis=0)
            grad = self.cat_intercept_coef(intercept_grad, grad)

        return (1 / self.X.shape[0]) * grad

    def grad_at_coef_eq0(self):
        """
        Computes the gradient when the coefficeint is zero and the intercept is set to the minimizer of the loss function when the coefficient is held at zero.
        """
        zero = self.get_zero_coef()

        if self.fit_intercept:
            intercept = self.intercept_at_coef_eq0()
            coef = self.cat_intercept_coef(intercept, zero)
            grad = self.grad(coef)
            return grad[1:]  # ignore the gradient of the intercept

        else:
            return self.grad(zero)

    def intercept_at_coef_eq0(self):
        raise NotImplementedError

    def default_init(self):
        coef = self.get_zero_coef()

        if self.fit_intercept:
            try:
                intercept = self.intercept_at_coef_eq0()
            except NotImplementedError:
                intercept = np.zeros(self.intercept_shape_)
                if len(intercept) == 1:
                    intercept = float(intercept)

            return self.cat_intercept_coef(intercept, coef)
        else:
            return coef


class GlmMultiResp(Glm):

    def get_z(self, x):
        return safe_data_mat_coef_mat_dot(X=self.X,
                                          coef=x.reshape(self.var_shape_),
                                          fit_intercept=self.fit_intercept)

    def cat_intercept_coef(self, intercept, coef):
        if intercept.ndim == 1:
            intercept = intercept.reshape(1, -1)
        return np.vstack([intercept, coef])
