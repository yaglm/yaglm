import numpy as np
from ya_glm.opt.glm_loss.base import Glm, GlmMultiResp
from ya_glm.opt.glm_loss.utils import safe_covar_mat_op_norm


def sample_losses(z, y):
    return 0.5 * (z - y) ** 2


def sample_losses_multi_resp(z, y):
    return sample_losses(z, y).sum(axis=1)


def sample_grads(z, y):
    return z - y


def compute_lip(X, fit_intercept=True, sample_weight=None):
    """
    Gets the gradient lipschitz constant for the linear regression loss funcion.

    Output
    ------
    lip: float
        The gradient Lipschitz constant
    """
    op_norm = safe_covar_mat_op_norm(X=X,
                                     fit_intercept=fit_intercept,
                                     sample_weight=sample_weight)

    return (1 / X.shape[0]) * op_norm ** 2


class LinReg(Glm):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    compute_lip = staticmethod(compute_lip)

    def intercept_at_coef_eq0(self):
        return np.average(self.y, weights=self.sample_weight)


class LinRegMultiResp(GlmMultiResp):
    sample_losses = staticmethod(sample_losses_multi_resp)
    sample_grads = staticmethod(sample_grads)
    compute_lip = staticmethod(compute_lip)

    def intercept_at_coef_eq0(self):
        return np.average(self.y, axis=0, weights=self.sample_weight)
