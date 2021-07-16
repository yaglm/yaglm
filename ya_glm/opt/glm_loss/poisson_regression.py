import numpy as np
from ya_glm.opt.glm_loss.base import Glm, GlmMultiResp


def sample_losses(z, y):
    return np.exp(z) - z * y


def sample_losses_multi_resp(z, y):
    return sample_losses(z, y).sum(axis=1)


def sample_grads(z, y):
    return np.exp(z) - y

# TODO: add exposure
class PoissonReg(Glm):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    def intercept_at_coef_eq0(self):
        # TODO: double check weighted case
        return np.average(self.y, weights=self.sample_weight)


class PoissonRegMultiResp(GlmMultiResp):
    sample_losses = staticmethod(sample_losses_multi_resp)
    sample_grads = staticmethod(sample_grads)

    def intercept_at_coef_eq0(self):
        # TODO: double check weighted case
        return np.average(self.y, axis=0, weights=self.sample_weight)
