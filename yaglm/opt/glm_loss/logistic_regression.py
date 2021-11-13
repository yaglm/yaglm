import numpy as np

from yaglm.opt.glm_loss.base import Glm, GlmInputLoss


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


class Logistic(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    # TODO: add this
    # sample_proxs = !!!!

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 0.25 / self.n_samples


class LogReg(Glm):

    GLM_LOSS_CLASS = Logistic

    def intercept_at_coef_eq0(self):
        # TODO: is this correct with the sample weights?
        return np.average(self.y, weights=self.sample_weight)
