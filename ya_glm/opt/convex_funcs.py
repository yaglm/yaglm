import numpy as np
from ya_glm.opt.base import Func

from ya_glm.linalg_utils import euclid_norm
from ya_glm.opt.prox import L2_prox, squared_l1_prox_pos


class L2Norm(Func):
    """
    f(x) = mult * ||x||_2
    """
    def __init__(self, mult=1.0):
        self.mult = mult

    def _eval(self, x):
        return self.mult * euclid_norm(x)

    def _prox(self, x, step):
        return L2_prox(x=x, mult=self.mult * step)

    @property
    def is_smooth(self):
        return False


class SquaredL1(Func):
    """
    The squared L1 norm also known as the exclusive Lasso
    f(x) = mult * (sum_i w_i |x_i|)^2

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.

    References
    ----------
    Lin, M., Sun, D., Toh, K.C. and Yuan, Y., 2019. A dual Newton based preconditioned proximal point algorithm for exclusive lasso models. arXiv preprint arXiv:1902.00151.
    """
    def __init__(self, mult=1.0, weights=None):
        self.mult = mult
        self.weights = weights

    def _eval(self, x):

        if self.weights is None:
            L1_norm_val = abs(x).sum()
        else:
            L1_norm_val = self.weights.ravel().T @ abs(x)

        return self.mult * L1_norm_val ** 2

    def _prox(self, x, step):
        return np.sign(x) * squared_l1_prox_pos(x=abs(x),
                                                step=self.mult * step,
                                                weights=self.weights)

    @property
    def is_smooth(self):
        return False
