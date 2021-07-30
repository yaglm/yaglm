import numpy as np
from ya_glm.opt.glm_loss.base import Glm
from ya_glm.opt.glm_loss.utils import safe_covar_mat_op_norm

from ya_glm.class_weight import get_sample_weight_balanced_classes


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


def sample_losses(z, y):
    return (1 - y) * z - logsig(z)


def sample_grads(z, y):
    """
    Compute sigmoid(z) - y component-wise, see http://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    idx = z < 0
    out = np.zeros_like(z)
    exp_z = np.exp(z[idx])
    y_idx = y[idx]
    out[idx] = ((1 - y_idx) * exp_z - y_idx) / (1 + exp_z)
    exp_nx = np.exp(-z[~idx])
    y_nidx = y[~idx]
    out[~idx] = ((1 - y_nidx) - y_nidx * exp_nx) / (1 + exp_nx)
    return out


def compute_lip(X, fit_intercept=True, sample_weight=None):

    op_norm = safe_covar_mat_op_norm(X=X,
                                     fit_intercept=fit_intercept,
                                     sample_weight=sample_weight)

    return (0.25/X.shape[0]) * op_norm ** 2


class LogReg(Glm):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    compute_lip = staticmethod(compute_lip)

    def __init__(self, X, y, fit_intercept=True, sample_weight=None,
                 balence_classes=False):

        if balence_classes:
            sample_weight = \
                 get_sample_weight_balanced_classes(y=y,
                                                    sample_weight=sample_weight)

        super().__init__(X=X, y=y, fit_intercept=fit_intercept,
                         sample_weight=sample_weight)

    def intercept_at_coef_eq0(self):
        # TODO: is this correct with the sample weights?
        return np.average(self.y, weights=self.sample_weight)
