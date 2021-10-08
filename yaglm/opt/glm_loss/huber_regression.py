import numpy as np
from scipy.optimize import root_scalar

from yaglm.opt.glm_loss.base import Glm, GlmMultiResp, GlmInputLoss
from yaglm.opt.utils import safe_vectorize
from yaglm.opt.glm_loss.linear_regression import \
    compute_lip as lin_reg_compute_lip


def huber_eval_1d(r, knot=1):
    x_abs = abs(r)
    if x_abs <= knot:
        return 0.5 * r ** 2
    else:
        return knot * (x_abs - 0.5 * knot)


vec_huber_eval = safe_vectorize(huber_eval_1d)


def huber_grad_1d(r, knot=1):

    if abs(r) <= knot:
        return r
    else:
        return knot * np.sign(r)


vec_huber_grad = safe_vectorize(huber_grad_1d)


def huber_prox_1d(z, y, knot=1, step=1):
    # https://math.stackexchange.com/questions/1650411/proximal-operator-of-the-huber-loss-function
    # also see (2.2) or https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

    r = z - y
    # p = r - step * r / max(abs(r), step + 1)
    # return p + y
    return z - step * r / max(abs(r), step + 1)


vec_huber_prox = safe_vectorize(huber_prox_1d)


def sample_losses(z, y, knot=1):
    return vec_huber_eval(r=z - y, knot=knot)


def sample_losses_multi_resp(z, y, knot=1):
    return vec_huber_eval(r=z - y, knot=knot).sum(axis=1)


def sample_grads(z, y, knot=1):
    return vec_huber_grad(r=z - y, knot=knot)


def huberized_mean_1d(values, knot=1, sample_weight=None, **kws):
    """
    Computes the huberized mean of a set of 1d samples.

    Parameters
    ----------
    values: array-like, (n_samples)

    knot: float
        Where the knot is.

    sample_weight: None, array-like (n_samples, )

    **kws:
        keyword arguments to scipy.optimize.root_scalar

    Output
    ------
    avg: float
    """

    def score(x):
        scores = np.array([huber_grad_1d(r=x - v, knot=knot)
                           for v in values.ravel()])
        return np.average(scores, weights=sample_weight)

    avg = np.average(values, weights=sample_weight)
    med = np.median(values)

    bracket = [min(values), max(values)]

    result = root_scalar(f=score, bracket=bracket,
                         x0=avg, x1=med,
                         **kws)

    huber_mean = result.root
    return huber_mean


def huberized_mean(values, axis=0, knot=1, sample_weight=None, **kws):
    """
    Computes the huberized mean along the axis of an array.

    Parameters
    ----------
    values: array-like, (n_samples)

    knot: float
        Where the knot is.

    sample_weight: None, array-like (n_samples, )

    **kws:
        keyword arguments to scipy.optimize.root_scalar

    Output
    ------
    avg: array-like or float
    """
    out = np.apply_along_axis(func1d=huberized_mean_1d,
                              arr=values, axis=axis,
                              knot=knot, sample_weight=sample_weight, **kws)
    if out.ndim == 0:
        out = float(out)
    return out


def compute_lip(knot=1.35, **kws):
    return lin_reg_compute_lip(**kws)


class Huber(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    # TODO: add this
    # sample_proxs = !!!!

    @property
    def is_smooth(self):
        return self.loss_kws['knot'] > 0


class HuberReg(Glm):

    GLM_LOSS_CLASS = Huber
    compute_lip = staticmethod(compute_lip)

    def __init__(self, X, y, fit_intercept=True, sample_weight=None,
                 knot=1.35):

        super().__init__(X=X, y=y, fit_intercept=fit_intercept,
                         sample_weight=sample_weight, knot=knot)

    def intercept_at_coef_eq0(self):
        return huberized_mean(values=self.y,
                              axis=0,
                              sample_weight=self.sample_weight,
                              knot=self.loss_kws['knot'])


class HuberMulti(GlmInputLoss):
    sample_losses = staticmethod(sample_losses_multi_resp)
    sample_grads = staticmethod(sample_grads)
    sample_proxs = staticmethod(vec_huber_prox)

    @property
    def is_smooth(self):
        return self.loss_kws['knot'] > 0


class HuberRegMultiResp(GlmMultiResp):

    GLM_LOSS_CLASS = HuberMulti
    compute_lip = staticmethod(compute_lip)

    def __init__(self, X, y, fit_intercept=True, sample_weight=None,
                 knot=1.35):

        super().__init__(X=X, y=y, fit_intercept=fit_intercept,
                         sample_weight=sample_weight, knot=knot)

    def intercept_at_coef_eq0(self):
        return huberized_mean(values=self.y,
                              axis=0,
                              sample_weight=self.sample_weight,
                              knot=self.loss_kws['knot'])
