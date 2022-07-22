import numpy as np

from yaglm.opt.glm_loss.base import GlmInputLoss, Glm
from yaglm.opt.utils import safe_vectorize


def hinge(p):
    return np.maximum(0, 1 - p)


def hinge_prox_1d(p, step=1):
    """
    See http://proximity-operator.net/scalarfunctions.html
    or
    https://www.control.lth.se/fileadmin/control/staff/PontusGiselsson/DLandGANs/lec2.pdf
    """
    if p >= 1:
        return p
    elif p <= 1 - step:
        return p + step
    else:
        return 1


hinge_prox = safe_vectorize(hinge_prox_1d)


def sample_losses(z, y):
    return hinge(y * z)


# def sample_losses_multi_resp(z, y):
#     return sample_losses(z, y).sum(axis=1)


def sample_proxs(z, y, step=1):
    return (1/y) * hinge_prox(y * z, step=step)


def sample_grads(z, y):
    p = y * z
    g = -1 * (p <= 0).astype(float)
    return y * g


class Hinge(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    sample_proxs = staticmethod(sample_proxs)

    @property
    def is_smooth(self):
        return False

    @property
    def grad_lip(self):
        return None


class HingeReg(Glm):

    GLM_LOSS_CLASS = Hinge

    def intercept_at_coef_eq0(self):
        # values = self.y if self.offsets is None else self.y - self.offsets
        # TODO: we should be able to calculate this
        raise NotImplementedError()
