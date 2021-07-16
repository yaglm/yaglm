import numpy as np
from ya_glm.opt.base import Func
from ya_glm.opt.utils import euclid_norm


class L2Penalty(Func):
    """
    f(x) = mult * ||x||_2
    """
    def __init__(self, mult=1.0):
        self.mult = mult

    def _eval(self, x):
        return euclid_norm(x)

    def _prox(self, x, step):
        return L2_prox(x=x, mult=self.mult * step)


class GroupLasso(Func):
    """
    f(x) = mult * ||x||_2

    or

    f(x) = mult * sum_{g in groups} weights_g ||x_g||_2

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, groups, mult=1.0, weights=None):

        self.groups = groups

        if weights is None:
            self.pen_funcs = [L2Penalty(mult=mult)
                              for g in range(len(groups))]
        else:
            self.pen_funcs = [L2Penalty(mult=mult * weights[g])
                              for g in range(len(groups))]

    def _eval(self, x):
        return sum(self.pen_funcs[g]._eval(x[grp_idxs])
                   for g, grp_idxs in enumerate(self.groups))

    def _prox(self, x, step):

        out = np.zeros_like(x)

        for g, grp_idxs in enumerate(self.groups):
            # prox of group
            p = self.pen_funcs[g]._prox(x[grp_idxs], step=step)

            # put entries back into correct place
            for p_idx, x_idx in enumerate(grp_idxs):
                out[x_idx] = p[p_idx]

        return out


def L2_prox(x, mult):
    """
    Computes the proximal operator of mutl * ||x||_2
    """
    norm = euclid_norm(x)

    if norm <= mult:
        return np.zeros_like(x)
    else:
        return x * (1 - (mult / norm))
