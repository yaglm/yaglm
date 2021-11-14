from scipy.sparse import diags
import numpy as np

from yaglm.opt.base import Func
from yaglm.opt.utils import safe_data_mat_coef_dot, safe_data_mat_coef_mat_dot
from yaglm.opt.glm_loss.utils import safe_covar_mat_op_norm


# TODO how to handle loss kws
# currently pass them in as a dict but this seems wrong e.g.
# they should be arguments to init

class GlmInputLoss(Func):
    """
    Represents f(z) = (1/n_samples) sum_i w_i L(z_i, y_i) where L(z, y) is a loss function.
    """

    # subclasses should implement these
    sample_losses = None
    sample_grads = None
    sample_proxs = None

    def __init__(self, y, sample_weight=None, **loss_kws):
        self.y = y

        self.sample_weight = sample_weight
        self.n_samples = y.shape[0]
        self.loss_kws = loss_kws

    def _eval(self, x):
        losses = self.sample_losses(z=x, y=self.y, **self.loss_kws)

        if self.sample_weight is None:
            return losses.sum() / self.n_samples

        else:
            return (losses.T @ self.sample_weight) / self.n_samples

    def _grad(self, x):

        grads = self.sample_grads(z=x, y=self.y, **self.loss_kws)

        # possibly reweight
        if self.sample_weight is not None:
            grads = diags(self.sample_weight) @ grads

        grads /= self.n_samples
        return grads

    def _prox(self, x, step=1):
        if self.sample_weight is not None:
            raise NotImplementedError  # TODO

        return self.sample_proxs(z=x, y=self.y,
                                 step=step / self.n_samples,
                                 **self.loss_kws)

    @property
    def is_proximable(self):
        # if we have implemented the sample proxex then
        # this should be proximable
        return self.sample_proxs is not None


class Glm(Func):
    """
    Represents a GLM loss function.
    (1/n) sum_{i=1}^n w_i L(x_i^T coef + intercept, y_i)
    """

    # the GLM input loss
    # subclass needs to overwrite
    # TODO: is this the best way to do this?
    GLM_LOSS_CLASS = None

    # a method to compute the gradient Lipschitz constant
    compute_lip = None

    def __init__(self, X, y, fit_intercept=True, sample_weight=None,
                 **loss_kws):

        # instantiate GLM input loss class
        self.glm_loss = self.GLM_LOSS_CLASS(y=y,
                                            sample_weight=sample_weight,
                                            **loss_kws)
        self.X = X
        self.fit_intercept = fit_intercept
        self._set_shape_data()

    @property
    def is_smooth(self):
        return self.glm_loss.is_smooth

    @property
    def is_proximable(self):
        return False

    @property
    def y(self):
        return self.glm_loss.y

    @property
    def sample_weight(self):
        return self.glm_loss.sample_weight

    @property
    def loss_kws(self):
        return self.glm_loss.loss_kws

    @property
    def grad_lip(self):
        if not hasattr(self, '_grad_lip'):

            input_loss_lip = self.glm_loss.grad_lip

            if input_loss_lip is not None:
                X_op_norm = \
                    safe_covar_mat_op_norm(X=self.X,
                                           fit_intercept=self.fit_intercept)

                self._grad_lip = input_loss_lip * X_op_norm ** 2
            else:
                self._grad_lip = None

        return self._grad_lip

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

    def get_z(self, x):
        return safe_data_mat_coef_dot(X=self.X, coef=x,
                                      fit_intercept=self.fit_intercept)

    def cat_intercept_coef(self, intercept, coef):
        return np.concatenate([[intercept], coef])

    def get_zero_coef(self):
        return np.zeros(self.coef_shape_)

    def _eval(self, x):
        return self.glm_loss.eval(self.get_z(x))

    def _grad(self, x):
        # compute grad at each sample
        sample_grads = self.glm_loss.grad(self.get_z(x))

        # get coefficient gradients
        grad = self.X.T @ sample_grads

        # possibly add intercept to gradient
        if self.fit_intercept:
            intercept_grad = sample_grads.sum(axis=0)
            grad = self.cat_intercept_coef(intercept_grad, grad)

        return grad

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
