from itertools import chain
from ya_glm.opt.glm_loss.linear_regression import LinReg, LinRegMultiResp
from ya_glm.opt.glm_loss.huber_regression import HuberReg, HuberRegMultiResp
from ya_glm.opt.glm_loss.multinomial import Multinomial
from ya_glm.opt.glm_loss.poisson_regression import PoissonReg, \
    PoissonRegMultiResp
from ya_glm.opt.glm_loss.quantile_regression import QuantileReg, QuantileRegMultiResp
from ya_glm.opt.glm_loss.logistic_regression import LogReg

from ya_glm.opt.base import Func

_LOSS_CLS_VEC = {'lin_reg': LinReg,

                 'huber_reg': HuberReg,

                 'log_reg': LogReg,

                 'poisson': PoissonReg,

                 'quantile': QuantileReg,
                 }

_LOSS_CLS_MAT = {
                 'lin_reg': LinRegMultiResp,

                 'huber_reg': HuberRegMultiResp,

                 'multinomial': Multinomial,

                 'poisson': PoissonRegMultiResp,

                 'quantile': QuantileRegMultiResp,
                }

_LOSS_FUNC_CLS2STR = {v: k for (k, v) in chain(_LOSS_CLS_VEC.items(),
                                               _LOSS_CLS_MAT.items())}


def get_glm_loss(X, y,
                 loss_func='lin_reg', loss_kws={},
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

    fit_intercept: bool
        Whether or not to fit an intercept.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    loss_func: str
        Which GLM loss function to use.
        Must be one of ['linear_regression', 'logistic_regression'].
        This may also be an instance of ya_glm.opt.base.Func.

    precomp_lip: None, float
        (Optional) Precomputed Lipchitz constant

    Output
    ------
    glm_loss: ya_glm.opt.Func
        The GLM loss function object.
    """

    if isinstance(loss_func, Func):
        return loss_func

    if y.ndim == 1 or y.shape[1] == 1:
        CLS = _LOSS_CLS_VEC[loss_func]
    else:
        CLS = _LOSS_CLS_MAT[loss_func]

    kws = {'X': X, 'y': y, 'fit_intercept': fit_intercept,
           'loss_kws': loss_kws}

    if sample_weight is not None:
        kws['sample_weight'] = sample_weight

    return CLS(**kws)
