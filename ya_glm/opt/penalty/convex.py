import numpy as np
from scipy.linalg import svd

from ya_glm.opt.base import Func, EntrywiseFunc
from ya_glm.opt.convex_funcs import L2Norm, SquaredL1
from ya_glm.opt.prox import soft_thresh, L2_prox
from ya_glm.linalg_utils import euclid_norm, leading_sval


class Ridge(EntrywiseFunc):
    """
    f(x) = 0.5 * pen_val * sum_{j=1}^d weights_j x_j^2

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, pen_val=1.0, weights=None):

        self.pen_val = pen_val
        if weights is not None:
            weights = np.array(weights).reshape(-1)

        self.weights = weights

        if self.weights is None:
            self._grad_lip = pen_val
        else:
            self._grad_lip = pen_val * np.array(self.weights).max()

    def _eval(self, x):

        if self.weights is None:
            norm_val = (x ** 2).sum()

        else:
            norm_val = self.weights.T @ (x ** 2)

        return 0.5 * self.pen_val * norm_val

    def _prox(self, x, step):

        # set shrinkage values
        if self.weights is None:
            shrink_vals = step * self.pen_val
        else:
            shrink_vals = (step * self.pen_val) * self.weights

        return x / (1 + shrink_vals)

    def _grad(self, x):
        coef_grad = x
        if self.weights is not None:
            coef_grad = coef_grad * self.weights
        return self.pen_val * coef_grad

    @property
    def is_smooth(self):
        return True


class GeneralizedRidge(Func):
    """
    f(x) = 0.5 * pen_val * ||mat @ x ||_2^2

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    mat: None, array-like
        The matrix transform.
    """
    def __init__(self, pen_val=1.0, mat=None):

        self.pen_val = pen_val
        self.mat = mat

        if mat is None:
            self._grad_lip = pen_val
        else:
            # TODO: double check
            self._grad_lip = pen_val * leading_sval(mat) ** 2

            # cache this for gradient computations
            # TODO: get this to work with sparse matrices
            # TODO: prehaps allow precomputed mat_T_mat
            self.mat_T_mat = self.mat.T @ self.mat

    def _eval(self, x):

        if self.mat is None:
            norm_val = (x ** 2).sum()

        else:
            norm_val = ((self.mat @ x) ** 2).sum()

        return 0.5 * self.pen_val * norm_val

    # def _prox(self, x, step):
    # TODO: think about this

    def _grad(self, x):
        grad = x
        if self.mat is not None:
            grad = self.mat_T_mat @ grad

        return self.pen_val * grad

    @property
    def is_smooth(self):
        return True


class Lasso(EntrywiseFunc):
    """
    f(x) = pen_val * sum_{j=1}^d weights_j |x_j|

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) variable weights.
    """
    def __init__(self, pen_val=1.0, weights=None):

        self.pen_val = pen_val
        if weights is not None:
            weights = np.array(weights).reshape(-1)
        self.weights = weights

    def _eval(self, x):

        if self.weights is None:
            norm_val = abs(x).sum()

        else:
            norm_val = self.weights.T @ abs(x)

        return norm_val * self.pen_val

    def _prox(self, x, step):

        # set thresholding values
        if self.weights is None:
            thresh_vals = step * self.pen_val
        else:
            thresh_vals = (step * self.pen_val) * np.array(self.weights)

        # apply soft thresholding
        return soft_thresh(x, thresh_vals)

    @property
    def is_smooth(self):
        return False


class GroupLasso(Func):
    """
    f(x) = pen_val * ||x||_2

    or

    f(x) = pen_val * sum_{g in groups} weights_g ||x_g||_2

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) group weights.
    """
    def __init__(self, groups, pen_val=1.0, weights=None):

        self.groups = groups

        if weights is None:
            self.pen_funcs = [L2Norm(mult=pen_val)
                              for g in range(len(groups))]
        else:
            self.pen_funcs = [L2Norm(mult=pen_val * weights[g])
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

    @property
    def is_smooth(self):
        return False


class ExclusiveGroupLasso(Func):
    """
    The exclusive group Lasso

    f(x) = pen_val * sum_{g in groups} (sum_{i in g} w_i |x_i|)^2

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) feature weights.

    References
    ----------
    Lin, M., Sun, D., Toh, K.C. and Yuan, Y., 2019. A dual Newton based preconditioned proximal point algorithm for exclusive lasso models. arXiv preprint arXiv:1902.00151.

    Campbell, F. and Allen, G.I., 2017. Within group variable selection through the exclusive lasso. Electronic Journal of Statistics, 11(2), pp.4220-4257.
    """
    def __init__(self, groups, pen_val=1.0):

        self.groups = groups
        self.pen_funcs = [SquaredL1(mult=pen_val)
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

    @property
    def is_smooth(self):
        return False


class NuclearNorm(Func):
    # https://github.com/scikit-learn-contrib/lightning/blob/master/lightning/impl/penalty.py

    def __init__(self, pen_val=1, weights=None):
        self.pen_val = pen_val
        if weights is not None:
            weights = np.array(weights).ravel()

        self.weights = weights

    def _prox(self, x, step=1):

        U, s, V = svd(x, full_matrices=False)
        if self.weights is None:
            thresh = self.pen_val * step
        else:
            thresh = (self.pen_val * step) * self.weights

        s = np.maximum(s - thresh, 0)
        U *= s
        return np.dot(U, V)

    def _eval(self, x):

        U, s, V = svd(x, full_matrices=False)
        if self.weights is None:
            return self.pen_val * np.sum(s)
        else:
            return self.pen_val * self.weights.T @ s

    @property
    def is_smooth(self):
        return False


class MultiTaskLasso(Func):
    def __init__(self, pen_val=1, weights=None):
        self.pen_val = pen_val
        self.weights = weights

    def _eval(self, x):

        if self.weights is None:
            return self.pen_val * sum(euclid_norm(x[r, :])
                                      for r in range(x.shape[0]))

        else:
            return self.pen_val * sum(self.weights[r] * euclid_norm(x[r, :])
                                      for r in range(x.shape[0]))

    def _prox(self, x, step=1):
        out = np.zeros_like(x)
        for r in range(x.shape[0]):
            if self.weights is None:
                m = self.pen_val * step
            else:
                m = self.pen_val * step * self.weights[r]

            out[r, :] = L2_prox(x[r, :], mult=m)

        return out

    @property
    def is_smooth(self):
        return False


class GeneralizedLasso(Func):
    """
    f(x) = pen_val * ||mat @ x ||_1

    or

    pen_val * sum_j weights_j |mat[j, :].T @ x|

    Parameters
    ----------
    pen_val: float
        The multiplicative penalty value.

    mat: None, array-like
        The matrix transformation
    """
    def __init__(self, pen_val=1.0, mat=None, weights=None):

        self.mat = mat
        self.lasso = Lasso(pen_val=pen_val, weights=weights)

    @property
    def is_smooth(self):
        return False

    def _eval(self, x):
        if self.mat is None:
            z = x
        else:
            z = self.mat @ x

        return self.lasso._eval(z)
