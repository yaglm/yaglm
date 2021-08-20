import numpy as np
from ya_glm.opt.base import Func


class GroupExclusiveLasso(Func):
    """
    The group exclusive Lasso

    f(x) = mult * sum_{g in groups} (sum_{i in g} w_i |x_i|)^2

    Parameters
    ----------
    mult: float
        The multiplicative penalty value.

    weights: None, array-like
        The (optional) feature weights.

    References
    ----------
    Lin, M., Sun, D., Toh, K.C. and Yuan, Y., 2019. A dual Newton based preconditioned proximal point algorithm for exclusive lasso models. arXiv preprint arXiv:1902.00151.

    Campbell, F. and Allen, G.I., 2017. Within group variable selection through the exclusive lasso. Electronic Journal of Statistics, 11(2), pp.4220-4257.
    """
    def __init__(self, groups, mult=1.0, weights=None):

        self.groups = groups

        if weights is None:
            self.pen_funcs = [SquaredL1(mult=mult)
                              for g in range(len(groups))]
        else:
            self.pen_funcs = [SquaredL1(mult=mult, weights=weights[grp_idxs])
                              for g, grp_idxs in enumerate(self.groups)]

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


def squared_l1_prox_pos(x, step=1, weights=None, check=False):
    """
    prox_{step * f}(x) for postive vectors x

    where f(z) = (sum_i w_i |z_i|)^2

    Parameters
    ----------
    x: array-like
        The vector to evaluate the prox at. Note this must be positive.

    step: float
        The prox step size.

    weights: array-like
        The (optional) positive weights.

    check: bool
        Whether or not to check that x is non-negative.

    Output
    ------
    p: array-like
        The value of the proximal operator.

    References
    ----------
    Lin, M., Sun, D., Toh, K.C. and Yuan, Y., 2019. A dual Newton based preconditioned proximal point algorithm for exclusive lasso models. arXiv preprint arXiv:1902.00151.
    """
    if check:
        assert all(x >= 0)

    if weights is None:
        weights = np.ones_like(x)

    # get indices to sort x / weights in decreasing order
    x_over_w = x / weights
    decr_sort_idxs = np.argsort(x_over_w)[::-1]

    x_sort = x[decr_sort_idxs]
    weights_sort = weights[decr_sort_idxs]

    # compute threshold value
    s = np.cumsum(x_sort * weights_sort)
    L = np.cumsum(weights_sort ** 2)
    thresh = max(s / (1 + 2 * step * L))

    # return soft thresholding
    return np.maximum(x - thresh, 0)
