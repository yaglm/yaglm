import numpy as np
from ya_glm.opt.base import Func
from ya_glm.opt.glm_loss.base import Glm, GlmMultiResp
from ya_glm.opt.utils import euclid_norm, L2_prox


class L2Loss(Func):
    """
    f(r) = mult * (1/sqrt(n)) ||r||_2
    """

    def _eval(self, x):
        return (1 / np.sqrt(x.shape[0])) * euclid_norm(x)

    def _prox(self, x, step):
        return L2_prox(x=x, mult=(1 / np.sqrt(x.shape[0])) * step)


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
        return np.average(self.y, weights=self.sample_weight)


# TODO: add multiple response version
class LinRegMultiResp(GlmMultiResp):
    raise NotImplementedError
