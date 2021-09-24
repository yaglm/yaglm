import numpy as np
from scipy.linalg import svd

from ya_glm.opt.base import Func
from ya_glm.linalg_utils import euclid_norm
from ya_glm.autoassign import autoassign


class CompositeGroup(Func):

    @autoassign
    def __init__(self, groups, func): pass

    @property
    def is_smooth(self):
        return False

    def eval(self, x):
        norms = np.array([euclid_norm(x[grp_idxs])
                         for grp_idxs in self.groups])
        return self.func.eval(norms)

    def _prox(self, x, step=1):
        # compute prox of the norms
        norms = np.array([euclid_norm(x[grp_idxs])
                         for grp_idxs in self.groups])
        norm_proxs = self.func.prox(norms, step=step)

        out = np.zeros_like(x)

        for g, grp_idxs in enumerate(self.groups):

            if norm_proxs[g] > np.finfo(float).eps:

                # group prox, if non-zero
                p = x[grp_idxs] * (norm_proxs[g] / norms[g])

                # put entries back into correct place
                for p_idx, x_idx in enumerate(grp_idxs):
                    out[x_idx] = p[p_idx]

        return out


class CompositeMultiTaskLasso(Func):

    @autoassign
    def __init__(self, func): pass

    @property
    def is_smooth(self):
        return False

    def _eval(self, x):
        norms = np.array([euclid_norm(x[r, :]) for r in range(x.shape[0])])
        return self.func.eval(norms)

    def _prox(self, x, step=1):

        # compute prox of the norms
        norms = np.array([euclid_norm(x[r, :]) for r in range(x.shape[0])])
        norm_proxs = self.func.prox(norms, step=step)

        out = np.zeros_like(x)
        for r in range(x.shape[0]):
            if norm_proxs[r] > np.finfo(float).eps:
                out[r, :] = x[r, :] * (norm_proxs[r] / norms[r])

        return out


class CompositeNuclearNorm(Func):
    # https://github.com/scikit-learn-contrib/lightning/blob/master/lightning/impl/penalty.py

    @autoassign
    def __init__(self, func): pass

    @property
    def is_smooth(self):
        return False

    def _prox(self, x, step=1):

        U, s, V = svd(x, full_matrices=False)
        s = self.func.prox(s, step=step)
        U *= s
        return np.dot(U, V)

    def _eval(self, x):
        U, s, V = svd(x, full_matrices=False)
        return self.func.eval(s)


class CompositeGeneralizedLasso(Func):

    def __init__(self, func, mat=None): pass

    @property
    def is_smooth(self):
        return False

    def _eval(self, x):
        if self.mat is None:
            z = x
        else:
            z = self.mat @ x

        return self.func.eval(z)
