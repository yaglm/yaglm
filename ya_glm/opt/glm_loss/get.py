from ya_glm.info import is_multi_response
from ya_glm.opt.glm_loss.linear_regression import LinReg, LinRegMultiResp
from ya_glm.opt.glm_loss.huber_regression import HuberReg, HuberRegMultiResp
from ya_glm.opt.glm_loss.multinomial import Multinomial
from ya_glm.opt.glm_loss.poisson_regression import PoissonReg, \
    PoissonRegMultiResp
from ya_glm.opt.glm_loss.quantile_regression import QuantileReg, QuantileRegMultiResp
from ya_glm.opt.glm_loss.logistic_regression import LogReg

from ya_glm.opt.base import Func

_LOSS_FUNC_STR2CLS = {'lin_reg': LinReg,
                      'lin_reg_mr': LinRegMultiResp,

                      'huber_reg': HuberReg,
                      'huber_reg_mr': HuberRegMultiResp,

                      'log_reg': LogReg,
                      'multinomial': Multinomial,

                      'poisson': PoissonReg,
                      'poisson_mr': PoissonRegMultiResp,

                      'quantile': QuantileReg,
                      'quantile_mr': QuantileRegMultiResp
                      }

_LOSS_FUNC_CLS2STR = {v: k for (k, v) in _LOSS_FUNC_STR2CLS.items()}


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

    assert loss_func in _LOSS_FUNC_STR2CLS.keys()
    obj_class = _LOSS_FUNC_STR2CLS[loss_func]

    kws = {'X': X, 'y': y, 'fit_intercept': fit_intercept,
           'loss_kws': loss_kws}

    if sample_weight is not None:
        kws['sample_weight'] = sample_weight

    return obj_class(**kws)


def safe_is_multi_response(loss_func):
    if isinstance(loss_func, Func):
        return is_multi_response(_LOSS_FUNC_CLS2STR[type(loss_func)])
    else:
        return is_multi_response(loss_func)
