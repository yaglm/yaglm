import numpy as np
from scipy.linalg import svd

from yaglm.opt.base import Func
from yaglm.linalg_utils import euclid_norm
from yaglm.autoassign import autoassign
from yaglm.opt.penalty.convex import Ridge


class CompositeL2Norm(Func):
    @autoassign
    def __init__(self, func): pass

    @property
    def is_smooth(self):
        return False

    @property
    def is_proximable(self):
        return self.func.is_proximable

    def _eval(self, x):
        return self.func.eval(euclid_norm(x))

    def _prox(self, x, step):
        norm = euclid_norm(x)
        norm_prox = self.func.prox(norm, step=step)
        if norm_prox > np.finfo(float).eps:
            return x * (norm_prox / norm)
        else:
            return np.zeros_like(x)


class CompositeGroup(Func):

    @autoassign
    def __init__(self, groups, func): pass

    @property
    def is_smooth(self):
        return False

    @property
    def is_proximable(self):
        return self.func.is_proximable

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

    @property
    def is_proximable(self):
        return self.func.is_proximable

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

    @property
    def is_proximable(self):
        return self.func.is_proximable

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

    @property
    def is_proximable(self):
        return False

    def _eval(self, x):
        if self.mat is None:
            z = x
        else:
            z = self.mat @ x

        return self.func.eval(z)

###########################
# ElasticNet like classes #
###########################


class CompositeWithRidgeMixin:
    """
    Represents the sum of some function with a ridge penalty.

    Attributes
    ----------
    func:

    ridge:
    """

    @property
    def is_smooth(self):
        return self.func.is_smooth

    @property
    def is_proximable(self):
        return self.func.is_proximable \
            and self.ridge.weights is not None  # TODO: see below

    def _eval(self, x):
        return self.func._eval(x) + self.ridge._eval(x)

    def _grad(self, x):
        return self.func._grad(x) + self.ridge._grad(x)

    def _prox(self, x, step):
        if self.ridge.weights is not None:
            # TODO: figure this out. Not sure if this formula works in general
            # for non-convex + weighted ridge
            raise NotImplementedError("Does the prox decomposition "
                                      "fomula work with non-convex plus "
                                      "weighted ridge?")

        y = self.func.prox(x, step=step)
        return self.ridge._prox(y, step=step)


class EntrywiseWithRidge(CompositeWithRidgeMixin, Func):
    def __init__(self, func, ridge_pen_val, ridge_weights=None):
        self.func = func
        self.ridge = Ridge(pen_val=ridge_pen_val, weights=ridge_weights)

    @property
    def is_proximable(self):
        return self.func.is_proximable

    def _prox(self, x, step):
        # the prox-decomposition formula works for weighted ridge
        # for entrywise penalties!
        y = self.func.prox(x, step=step)
        return self.ridge._prox(y, step=step)


class CompositeGroupWithRidge(CompositeWithRidgeMixin, Func):

    @autoassign
    def __init__(self, groups, func, ridge_pen_val, ridge_weights=None):
        self.func = CompositeGroup(groups=groups, func=func)
        self.ridge = Ridge(pen_val=ridge_pen_val, weights=ridge_weights)


class CompositeMultiTaskLassoWithRidge(CompositeWithRidgeMixin, Func):

    def __init__(self, func, ridge_pen_val, ridge_weights=None):
        self.func = CompositeMultiTaskLasso(func=func)
        self.ridge = Ridge(pen_val=ridge_pen_val, weights=ridge_weights)
