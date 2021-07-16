import numpy as np
from scipy.linalg import svd

from ya_glm.opt.base import Func
from ya_glm.opt.utils import decat_coef_inter_mat, euclid_norm
from ya_glm.opt.penalty.GroupLasso import L2_prox


class MatWithIntercept(Func):
    def __init__(self, func):
        self.func = func

    def eval(self, x):
        coef, _ = decat_coef_inter_mat(x)
        return self.func.eval(coef)

    def grad(self, x):
        coef, _ = decat_coef_inter_mat(x)
        g = self.func.grad(coef)
        g = np.vstack([np.zeros(x.shape[1]), g])
        return g

    def prox(self, x, step=1):
        coef, intercept = decat_coef_inter_mat(x)
        p = self.func.prox(coef, step)
        p = np.vstack([intercept, p])
        return p

    @property
    def grad_lip(self):
        return self.func.grad_lip


class MatricizeEntrywisePen(Func):

    def __init__(self, func):
        self.func = func

    def eval(self, x):
        return self.func.eval(x.reshape(-1))

    def grad(self, x):
        g = self.func.grad(x.reshape(-1))
        return g.reshape(x.shape)

    def prox(self, x, step=1):
        p = self.func.prox(x.reshape(-1), step=step)
        return p.reshape(x.shape)

    @property
    def grad_lip(self):
        return self.func.grad_lip


class MultiTaskLasso(Func):
    def __init__(self, mult=1, weights=None):
        self.mult = mult
        self.weights = weights

    def _eval(self, x):

        if self.weights is None:
            return self.mult * sum(euclid_norm(x[r, :])
                                   for r in range(x.shape[0]))

        else:
            return self.mult * sum(self.weights[r] * euclid_norm(x[r, :])
                                   for r in range(x.shape[0]))

    def _prox(self, x, step=1):
        out = np.zeros_like(x)
        for r in range(x.shape[0]):
            if self.weights is None:
                m = self.mult * step
            else:
                m = self.mult * step * self.weights[r]

            out[r, :] = L2_prox(x[r, :], mult=m)

        return out


class NuclearNorm(Func):
    # https://github.com/scikit-learn-contrib/lightning/blob/master/lightning/impl/penalty.py

    def __init__(self, mult=1, weights=None):
        self.mult = mult
        if weights is not None:
            weights = np.array(weights).ravel()

        self.weights = weights

    def _prox(self, x, step=1):

        U, s, V = svd(x, full_matrices=False)
        if self.weights is None:
            thresh = self.mult * step
        else:
            thresh = (self.mult * step) * self.weights

        s = np.maximum(s - thresh, 0)
        U *= s
        return np.dot(U, V)

    def _eval(self, x):

        U, s, V = svd(x, full_matrices=False)
        if self.weights is None:
            return self.mult * np.sum(s)
        else:
            return self.mult * self.weights.T @ s
