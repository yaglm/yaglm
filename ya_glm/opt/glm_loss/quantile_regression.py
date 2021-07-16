import numpy as np
from scipy.optimize import minimize_scalar
from ya_glm.opt.glm_loss.base import Glm, GlmMultiResp
from ya_glm.opt.utils import safe_vectorize


def tilted_L1(u, quantile=0.5):
    """
    tilted_L1(u; quant) = quant * [u]_+ + (1 - quant) * [u]_
    """
    return 0.5 * abs(u) + (quantile - 0.5) * u


def tilted_L1_prox_1d(x, step, quantile=0.5):
    """
    prox(x) = argmin_z rho_quantile(z) + (0.5 / step) * ||x - z||_2^2

    See Lemma 1 of ADMM for High-Dimensional Sparse Penalized
Quantile Regression
    """
    if step < np.finfo(float).eps:
        return 0

    t_a = quantile * step  # tau / alpha

    if x > t_a:
        return x - t_a

    t_m1_a = (quantile - 1) * step
    if t_m1_a <= x:
        return 0

    else:
        return x - t_m1_a


tilted_L1_prox = safe_vectorize(tilted_L1_prox_1d)


def _tilted_L1_grad_1d(x, quantile=0.5):
    if x == 0:
        return 0

    elif x < 0:
        return - (1 - quantile)

    else:
        return quantile


tilted_L1_grad = safe_vectorize(_tilted_L1_grad_1d)


def weighted_quantile_1d(values, q=0.5,  sample_weight=None, **kws):
    if sample_weight is None:
        return np.quantile(a=values, q=q)

    def loss(x):
        values = np.array([tilted_L1(v, quantile=q)
                           for v in values.ravel()])

        return np.average(values, weights=sample_weight)

    bracket = [min(values), max(values)]

    result = minimize_scalar(f=loss, bracket=bracket, **kws)

    return result.x


def weighted_quantile(values, q=0.5, axis=0, sample_weight=None, **kws):
    if sample_weight is None:
        return np.quantile(a=values, q=q, axis=axis)
    else:
        out = np.apply_along_axis(func1d=weighted_quantile_1d,
                                  arr=values, axis=axis,
                                  q=q,
                                  sample_weight=sample_weight, **kws)
        if out.ndim == 0:
            out = float(out)
        return out


def sample_losses(z, y, quantile=0.5):
    return tilted_L1(z - y, quantile=quantile)


def sample_losses_multi_resp(z, y):
    return sample_losses(z, y).sum(axis=1)


def sample_grads(z, y, quantile=0.5):
    return tilted_L1_grad(z - y, quantile=quantile)


class QuantileMixin:
    @property
    def quantile(self):
        return self.loss_kws.get('quantile', 0.5)

    def intercept_at_coef_eq0(self):
        return weighted_quantile(values=self.y,
                                 axis=0,
                                 sample_weight=self.sample_weight,
                                 q=self.quantile)


class QuantileReg(QuantileMixin, Glm):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)


class QuantileRegMultiResp(QuantileMixin, GlmMultiResp):
    sample_losses = staticmethod(sample_losses_multi_resp)
    sample_grads = staticmethod(sample_grads)
