import numpy as np
from scipy.linalg import svd

from ya_glm.opt.base import Func
from ya_glm.linalg_utils import euclid_norm


class CompositeGroup(Func):

    def __init__(self, groups, func):

        self.func = func
        self.groups = groups

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

    def __init__(self, func):
        self.func = func

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

    def __init__(self, func):
        self.func = func

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
