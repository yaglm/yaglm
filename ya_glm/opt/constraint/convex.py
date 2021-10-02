import numpy as np
from sklearn.isotonic import isotonic_regression
from ya_glm.opt.base import Func


class Constraint(Func):

    def _eval(self, x):
        return 0

    @property
    def is_smooth(self):
        return False


class Positive(Constraint):

    def _prox(self, x, step=1):
        p = np.zeros_like(x)
        pos_mask = x > 0
        p[pos_mask] = x[pos_mask]
        return p

    @property
    def is_proximable(self):
        return True


class Simplex(Constraint):

    def __init__(self, mult=1):
        self.mult = mult

    def _prox(self, x, step=1):
        # TODO: z is what I think it is right?
        p = project_simplex(x.reshape(-1), z=self.mult)
        return p.reshape(x.shape)

    @property
    def is_proximable(self):
        return True


class L1Ball(Constraint):

    def __init__(self, mult=1):
        self.mult = mult

    def _prox(self, x, step=1):
        p = project_l1_ball(x.reshape(-1), z=self.mult)
        return p.reshape(x.shape)

    @property
    def is_proximable(self):
        return True


# See https://gist.github.com/mblondel/6f3b7aaad90606b98f71
# for more algorithms.
def project_simplex(v, z=1):
    # z is what the entries need to add up to, e.g. z=1 for probability simplex
    if np.sum(v) <= z:  # don't we want the simplex to mean sum == z not sum <= z?
        return v        # also this doesn't work when v has, say, all negative entries

    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w


def project_l1_ball(v, z=1):
    return np.sign(v) * project_simplex(np.abs(v), z)

class L2Ball(Constraint):

    def __init__(self, mult=1):
        assert mult > 0
        self.mult = mult

    def _prox(self, x, step=1):
        return x / np.max([np.linalg.norm(x)/self.mult, 1])

    @property
    def is_proximable(self):
        return True

class Isotonic(Constraint):
    """Constraint for x1 <= ... <= xn or
    x1 >= ... >= xn """
    # TODO: allow for general isotonic regression
    # where the order relations are a simple directed
    # graph. For an algorithm see Nemeth and Nemeth, "How to project onto an
    # isotone projection cone", JLLA 2010
    def __init__(self, increasing=True):
        self.increasing = increasing

    def _prox(self, x, step=1):
        # computes the projection of x onto the monotone cone
        # using the PAVA algorithm
        return isotonic_regression(x, increasing=self.increasing)

    @property
    def is_proximable(self):
        return True
