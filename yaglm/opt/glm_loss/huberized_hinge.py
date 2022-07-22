from yaglm.opt.glm_loss.base import GlmInputLoss, Glm
from yaglm.opt.utils import safe_vectorize


def _huberized_hinge_1d(p):
    if p >= 1:
        return 0
    elif p <= 0:
        return 0.5 - p
    else:
        return 0.5 * (1 - p)**2


huberized_hinge = safe_vectorize(_huberized_hinge_1d)


def _huberized_hinge_grad_1d(p):
    if p >= 1:
        return 0
    elif p <= 0:
        return -1
    else:
        return p - 1


huberized_hinge_grad = safe_vectorize(_huberized_hinge_grad_1d)


def sample_losses(z, y):
    return huberized_hinge(y * z)


# def sample_proxs(z, y, step=1):
#     raise NotImplementedError()

def sample_grads(z, y):
    return huberized_hinge_grad(y * z)


class HuberizedHinge(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 1 / self.n_samples


class HuberizedHingeReg(Glm):

    GLM_LOSS_CLASS = HuberizedHinge

    def intercept_at_coef_eq0(self):
        # values = self.y if self.offsets is None else self.y - self.offsets
        # TODO: we should be able to calculate this
        raise NotImplementedError()
