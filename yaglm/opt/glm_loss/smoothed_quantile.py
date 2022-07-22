import numpy as np
from scipy.special import expit


from yaglm.opt.glm_loss.base import GlmInputLoss, Glm
from yaglm.opt.glm_loss.quantile_regression import weighted_quantile


def smoothed_tilted_l1(u, quantile=0.5, smooth_param=0.5):
    # return quantile * u + smooth_param * np.log(1 + np.exp(-u/smooth_param))
    return quantile * u + smooth_param * np.logaddexp(0, -u/smooth_param)


def smoothed_tilted_l1_grad(u, quantile=0.5, smooth_param=0.5):
    # return quantile - (1 / (1 + np.exp(u / smooth_param)))
    return quantile - expit(- u / smooth_param)


def sample_losses(z, y, quantile=0.5, smooth_param=0.5):
    return smoothed_tilted_l1(y - z,
                              quantile=quantile,
                              smooth_param=smooth_param)


def sample_grads(z, y, quantile=0.5, smooth_param=0.5):
    return y * smoothed_tilted_l1_grad(y - z,
                                       quantile=quantile,
                                       smooth_param=smooth_param)


class SmoothedQuantile(GlmInputLoss):
    """
    References
    ----------
    Zheng, S., 2011. Gradient descent algorithms for quantile regression with smooth approximation. International Journal of Machine Learning and Cybernetics, 2(3), pp.191-207.
    """
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        return 0.25 / (self.loss_kws['smooth_param'] * self.n_samples)


class SmoothedQuantileReg(Glm):

    GLM_LOSS_CLASS = SmoothedQuantile

    def __init__(self, X, y,
                 fit_intercept=True, sample_weight=None, offsets=None,
                 quantile=0.5, smooth_param=0.5):

        super().__init__(X=X, y=y,
                         fit_intercept=fit_intercept,
                         sample_weight=sample_weight,
                         offsets=offsets,
                         quantile=quantile,
                         smooth_param=smooth_param)

    def intercept_at_coef_eq0(self):

        values = self.y if self.offsets is None else self.y - self.offsets

        # this is not exact
        # TODO; how to handle this case? warning? maybe rename to intercept_at_coef_eq0_guess?
        return weighted_quantile(values=values,
                                 axis=0,
                                 sample_weight=self.sample_weight,
                                 q=self.loss_kws['quantile'])
