import numpy as np
from scipy.special import expit

from yaglm.opt.glm_loss.base import GlmInputLoss, Glm


def sample_losses(z, y):
    p = y * z
    # return np.log(1 + np.exp(-p))
    return np.logaddexp(0, -p)


def sample_grads(z, y):
    p = y * z
    # g = - 1 / (1 + np.exp(p))
    g = - expit(-p)
    return g * y


class LogisticHinge(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    # sample_proxs = staticmethod(sample_proxs)

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 0.25 / self.n_samples


class LogisticHingeReg(Glm):

    GLM_LOSS_CLASS = LogisticHinge

    def intercept_at_coef_eq0(self):
        # values = self.y if self.offsets is None else self.y - self.offsets
        # TODO: we should be able to calculate this
        raise NotImplementedError()
