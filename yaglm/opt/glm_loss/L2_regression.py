import numpy as np
from yaglm.opt.base import Func
from yaglm.opt.glm_loss.base import Glm, GlmMultiResp
from yaglm.opt.prox import L2_prox
from yaglm.linalg_utils import euclid_norm


class L2Loss(Func):
    """
    f(z) = mult * (1/sqrt(n)) ||y - z||_2
    """

    def __init__(self, y, sample_weight=None, offsets=None):
        self.y = y
        self.sample_weight = sample_weight
        self.offsets = offsets
        if sample_weight is not None:
            raise NotImplementedError
        if offsets is not None:
            raise NotImplementedError

    def _eval(self, x):
        return (1 / np.sqrt(x.shape[0])) * euclid_norm(self.y - x)

    def _prox(self, x, step):
        r = x - self.y
        p = L2_prox(x=r, mult=(1 / np.sqrt(x.shape[0])) * step)
        return p + self.y

    @property
    def is_smooth(self):
        return False

# TODO: add
class L2LossMulti(Func):
    def __init__(self):
        raise NotImplementedError


class L2Reg(Glm):
    """
    Regression with the L2 norm loss
    (1/sqrt(n))||y - X @ coef + intercept||_2

    This is the loss function used in the square-root loss estimator.

    References
    ----------
    Belloni, A., Chernozhukov, V. and Wang, L., 2011. Square-root lasso: pivotal recovery of sparse signals via conic programming. Biometrika, 98(4), pp.791-806.
    """

    GLM_LOSS_CLASS = L2Loss

    def intercept_at_coef_eq0(self):
        if self.offsets is not None:
            raise NotImplementedError

        return np.average(self.y, weights=self.sample_weight)


# TODO: add multiple response version
class L2RegMultiResp(GlmMultiResp):
    def __init__(self):
        raise NotImplementedError
