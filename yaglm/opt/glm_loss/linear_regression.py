import numpy as np

from yaglm.opt.glm_loss.base import Glm, GlmInputLoss, GlmMultiResp


def sample_losses(z, y):
    return 0.5 * (z - y) ** 2


def sample_losses_multi_resp(z, y):
    return sample_losses(z, y).sum(axis=1)


def sample_grads(z, y):
    return z - y


def sample_proxs(z, y, step=1):
    """
    computes prox_(step * f)(z)
    where
    f(z) = 0.5 * (z - y) ** 2
    """
    return (z + step * y) / (1 + step)


class LeastSquares(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    sample_proxs = staticmethod(sample_proxs)

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 1 / self.n_samples


class LinReg(Glm):

    GLM_LOSS_CLASS = LeastSquares

    def intercept_at_coef_eq0(self):
        return np.average(self.y, weights=self.sample_weight)


class LeastSquaresMulti(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    sample_proxs = staticmethod(sample_proxs)

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 1 / self.n_samples


class LinRegMultiResp(GlmMultiResp):
    GLM_LOSS_CLASS = LeastSquaresMulti

    def intercept_at_coef_eq0(self):
        return np.average(self.y, axis=0, weights=self.sample_weight)
