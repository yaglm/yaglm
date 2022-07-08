import numpy as np
from yaglm.opt.glm_loss.base import Glm, GlmMultiResp, GlmInputLoss


def sample_losses(z, y):
    return np.exp(z) - z * y


def sample_losses_multi_resp(z, y):
    return sample_losses(z, y).sum(axis=1)


def sample_grads(z, y):
    return np.exp(z) - y


class Poisson(GlmInputLoss):

    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    # TODO: add this
    # sample_proxs = !!!!

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        # Poissson loss is not Lipschitz differentiable
        return None


class PoissonMulti(GlmInputLoss):

    sample_losses = staticmethod(sample_losses_multi_resp)
    sample_grads = staticmethod(sample_grads)

    # TODO: add this
    # sample_proxs = !!!!

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        # Poissson loss is not Lipschitz differentiable
        return None


# TODO: add exposure
class PoissonReg(Glm):

    GLM_LOSS_CLASS = Poisson

    def intercept_at_coef_eq0(self):
        if self.offsets is not None:
            raise NotImplementedError

        # TODO: double check weighted case
        return np.average(self.y, weights=self.sample_weight)


class PoissonRegMultiResp(GlmMultiResp):

    GLM_LOSS_CLASS = PoissonMulti

    def intercept_at_coef_eq0(self):
        if self.offsets is not None:
            raise NotImplementedError

        # TODO: double check weighted case
        return np.average(self.y, axis=0, weights=self.sample_weight)
