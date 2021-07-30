from itertools import chain
from ya_glm.opt.glm_loss.linear_regression import LinReg, LinRegMultiResp
from ya_glm.opt.glm_loss.huber_regression import HuberReg, HuberRegMultiResp
from ya_glm.opt.glm_loss.multinomial import Multinomial
from ya_glm.opt.glm_loss.poisson_regression import PoissonReg, \
    PoissonRegMultiResp
from ya_glm.opt.glm_loss.quantile_regression import QuantileReg, QuantileRegMultiResp
from ya_glm.opt.glm_loss.logistic_regression import LogReg

_LOSS_CLS_VEC = {'lin_reg': LinReg,

                 'huber': HuberReg,

                 'log_reg': LogReg,

                 'poisson': PoissonReg,

                 'quantile': QuantileReg,
                 }

_LOSS_CLS_MAT = {
                 'lin_reg': LinRegMultiResp,

                 'huber': HuberRegMultiResp,

                 'multinomial': Multinomial,

                 'poisson': PoissonRegMultiResp,

                 'quantile': QuantileRegMultiResp,
                }

_LOSS_FUNC_CLS2STR = {v: k for (k, v) in chain(_LOSS_CLS_VEC.items(),
                                               _LOSS_CLS_MAT.items())}


def get_glm_loss(X, y, loss,
                 fit_intercept=True,
                 sample_weight=None):
    """
    Returns an GLM loss function object.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, )
        The training response data.

    loss:
        A loss config objecet

    fit_intercept: bool
        Whether or not to fit an intercept.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    Output
    ------
    glm_loss: ya_glm.opt.Func
        The GLM loss function object.
    """

    if y.ndim == 1 or y.shape[1] == 1:
        # 1d output
        CLS = _LOSS_CLS_VEC[loss.name]
    else:
        # multiple response output
        CLS = _LOSS_CLS_MAT[loss.name]

    kws = {'X': X, 'y': y,
           'fit_intercept': fit_intercept, 'sample_weight': sample_weight,
           **loss.loss_kws}

    if sample_weight is not None:
        kws['sample_weight'] = sample_weight

    return CLS(**kws)
