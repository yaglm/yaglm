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
    groups: list of lists, None
        The indices of the groups. If None, then puts everything in one group.

    pen_val: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) group weights.
    """
    def __init__(self, groups, pen_val=1.0, weights=None):

        # if groups=None put everything ine one group
        if groups is None:
            groups = [...]
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
            if grp_idxs == ...:  # group of everything
                out = p
            else:
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
    groups: list of lists, None
        The indices of the groups. If None, then puts everything in one group.


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

        # if groups=None put everything ine one group
        if groups is None:
            groups = [...]
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
            if grp_idxs == ...:  # group of everything
                out = p
            else:
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

########################
# ElasticNet Penalties #
########################


class ElasticNetLikeMixinCooprativeProx:
    """
    Mixin for elastic net like functions that look like lasso + ridge where lasso is either the entrywise or a group lasso.

    For these functions we tend to have a nice formula for the prox, namely

    prox_{lasso + ridge}(x) = prox_{lasso}(prox_{ridge}(x))

    This "prox decomposition" formula applies to functions that "get along well" e.g. in the sense of Theorem 1 of (Yu, 2013). This formula does not hold in general so be careful about applying it!

    While the original references work with unweighted ridges, we can verify the formula holds  with a weighted ridge by Theorem 1 of (Yu, 2013). E.g. follow the short proof of Proposition 2.1 in (Zhang et al, 2020).


    Attributes
    ----------
    lasso: Func
        The lasso/group Lasso

    ridge: Func
        The ridge.

    References
    ----------
    Yu, Y., 2013, December. On decomposing the proximal map. In Proceedings of the 26th International Conference on Neural Information Processing Systems-Volume 1 (pp. 91-99).

    Zhang, Y., Zhang, N., Sun, D. and Toh, K.C., 2020. An efficient Hessian based algorithm for solving large-scale sparse group Lasso problems. Mathematical Programming, 179(1), pp.223-263.
    """

    @property
    def is_smooth(self):
        return False  # self.lasso.pen_val == 0

    def _eval(self, x):
        return self.lasso._eval(x) + self.ridge._eval(x)

    def _grad(self, x):
        return self.lasso._grad(x) + self.ridge._grad(x)

    def _prox(self, x, step):
        # prox decomposition formula! works for weighted ridges
        # and group lassos!
        y = self.ridge._prox(x, step=step)
        return self.lasso._prox(y, step=step)


class ElasticNet(Func):

    def __init__(self, pen_val=1, mix_val=0.5,
                 lasso_weights=None, ridge_weights=None):

        self.lasso = Lasso(pen_val=pen_val * mix_val,
                           weights=lasso_weights)

        self.ridge = Ridge(pen_val=pen_val * (1 - mix_val),
                           weights=ridge_weights)


class GroupElasticNet(ElasticNetLikeMixinCooprativeProx, Func):

    def __init__(self, groups=None, pen_val=1, mix_val=1,
                 lasso_weights=None, ridge_weights=None):

        self.lasso = GroupLasso(groups=groups,
                                pen_val=pen_val * mix_val,
                                weights=lasso_weights)

        self.ridge = Ridge(pen_val=pen_val * (1 - mix_val),
                           weights=ridge_weights)


class MultiTaskElasticNet(ElasticNetLikeMixinCooprativeProx, Func):

    def __init__(self, pen_val=1, mix_val=1,
                 lasso_weights=None, ridge_weights=None):

        self.lasso = MultiTaskLasso(pen_val=pen_val * mix_val,
                                    weights=lasso_weights)

        self.ridge = Ridge(pen_val=pen_val * (1 - mix_val),
                           weights=ridge_weights)


class SparseGroupLasso(Func):
    """

    References
    ----------
    Zhang, Y., Zhang, N., Sun, D. and Toh, K.C., 2020. An efficient Hessian based algorithm for solving large-scale sparse group Lasso problems. Mathematical Programming, 179(1), pp.223-263.
    """

    def __init__(self, groups=None, pen_val=1, mix_val=0.5,
                 sparse_weights=None, group_weights=None):

        self.sparse = Lasso(pen_val=pen_val * mix_val,
                            weights=sparse_weights)

        self.group = GroupLasso(groups=groups,
                                pen_val=pen_val * (1 - mix_val),
                                weights=group_weights)

    @property
    def is_smooth(self):
        return False

    def _eval(self, x):
        return self.sparse._eval(x) + self.group._eval(x)

    def _prox(self, x, step):
        # prox decomposition
        # Prop 2.1 from Zhang et al 2020 goes through with weights
        y = self.sparse._prox(x, step=step)
        return self.group._prox(y, step=step)
