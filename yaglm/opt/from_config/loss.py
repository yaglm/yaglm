from itertools import chain
from yaglm.opt.glm_loss.linear_regression import LinReg, LinRegMultiResp
from yaglm.opt.glm_loss.huber_regression import HuberReg, HuberRegMultiResp
from yaglm.opt.glm_loss.multinomial import Multinomial
from yaglm.opt.glm_loss.poisson_regression import PoissonReg, \
    PoissonRegMultiResp
from yaglm.opt.glm_loss.quantile_regression import QuantileReg, QuantileRegMultiResp
from yaglm.opt.glm_loss.logistic_regression import LogReg

from yaglm.opt.glm_loss.L2_regression import L2Reg, L2RegMultiResp

_LOSS_CLS_VEC = {'lin_reg': LinReg,

                 'huber': HuberReg,

                 'log_reg': LogReg,

                 'poisson': PoissonReg,

                 'quantile': QuantileReg,

                 'L2': L2Reg
                 }

_LOSS_CLS_MAT = {
                 'lin_reg': LinRegMultiResp,

                 'huber': HuberRegMultiResp,

                 'multinomial': Multinomial,

                 'poisson': PoissonRegMultiResp,

                 'quantile': QuantileRegMultiResp,

                 'L2': L2RegMultiResp
                }

_LOSS_FUNC_CLS2STR = {v: k for (k, v) in chain(_LOSS_CLS_VEC.items(),
                                               _LOSS_CLS_MAT.items())}


avail_loss_names = list(set(_LOSS_CLS_VEC.keys()).union(_LOSS_CLS_MAT.keys()))


def get_glm_loss_func(config, X, y,
                      fit_intercept=True,
                      sample_weight=None):
    """
    Returns an GLM loss function object.

    Parameters
    ----------
    config: LossCnfig
        A loss config objecet

    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, )
        The training response data.

    fit_intercept: bool
        Whether or not to fit an intercept.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    Output
    ------
    glm_loss: yaglm.opt.Func
        The GLM loss function object.
    """

    if config.name not in avail_loss_names:
        raise NotImplementedError("{} is not currently supported by "
                                  "yaglm.opt.glm_loss".
                                  format(config))

    if y.ndim == 1 or y.shape[1] == 1:
        # 1d output
        CLS = _LOSS_CLS_VEC[config.name]
    else:
        # multiple response output
        CLS = _LOSS_CLS_MAT[config.name]

    kws = {'X': X, 'y': y,
           'fit_intercept': fit_intercept, 'sample_weight': sample_weight,
           **config.get_func_params()}

    if sample_weight is not None:
        kws['sample_weight'] = sample_weight

    return CLS(**kws)
