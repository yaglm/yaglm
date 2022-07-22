from yaglm.opt.glm_loss.linear_regression import LeastSquares, \
    LeastSquaresMulti
from yaglm.opt.glm_loss.L2_regression import L2Loss, L2LossMulti
from yaglm.opt.glm_loss.huber_regression import Huber, HuberMulti
from yaglm.opt.glm_loss.poisson_regression import Poisson, PoissonMulti
from yaglm.opt.glm_loss.logistic_regression import Logistic
from yaglm.opt.glm_loss.multinomial import MultinomialLoss
from yaglm.opt.glm_loss.quantile_regression import Quantile, QuantileMulti
from yaglm.opt.glm_loss.smoothed_quantile import SmoothedQuantile
from yaglm.opt.glm_loss.huberized_hinge import HuberizedHinge
from yaglm.opt.glm_loss.logistic_hinge import LogisticHinge
from yaglm.opt.glm_loss.hinge import Hinge


def get_glm_input_loss(config, y, sample_weight=None, offsets=None):

    is_mr = y.ndim == 2 and y.shape[1] > 1
    kws = {'y': y}

    if sample_weight is not None:
        kws['sample_weight'] = sample_weight

    if offsets is not None:
        kws['offsets'] = offsets

    # linear regression
    if config.name == 'lin_reg':
        if is_mr:
            return LeastSquaresMulti(**kws)
        else:
            return LeastSquares(**kws)

    # poisson regression
    elif config.name == 'poisson':
        if is_mr:
            return PoissonMulti(**kws)
        else:
            return Poisson(**kws)

    # logistic regression
    elif config.name == 'log_reg':
        return Logistic(**kws)

    # multinomial regression
    elif config.name == 'multinomial':
        return MultinomialLoss(**kws)

    # huber regression
    elif config.name == 'huber':
        if is_mr:
            return HuberMulti(**kws, knot=config.knot)
        else:
            return Huber(**kws, knot=config.knot)

    # quantile regression
    elif config.name == 'quantile':
        if is_mr:
            return QuantileMulti(**kws, quantile=config.quantile)
        else:
            return Quantile(**kws, quantile=config.quantile)

    # smoothed quantile regression
    elif config.name == 'smoothed_quantile':
        if is_mr:
            raise NotImplementedError("TODO")
        else:
            return SmoothedQuantile(**kws,
                                    quantile=config.quantile,
                                    smooth_param=config.smooth_param)

    # L2 regression
    elif config.name == 'l2':
        if is_mr:
            return L2LossMulti(**kws)
        else:
            return L2Loss(**kws)

    # hinge
    elif config.name == 'hinge':
        if is_mr:
            raise NotImplementedError("TODO")
        else:
            return Hinge(**kws)

    # huberized hinge
    elif config.name == 'huberize_hinge':
        if is_mr:
            raise NotImplementedError("TODO")
        else:
            return HuberizedHinge(**kws)

    # logistic hinge
    elif config.name == 'logistic_hinge':
        if is_mr:
            raise NotImplementedError("TODO")
        else:
            return LogisticHinge(**kws)

    else:
        raise NotImplementedError("Could not find input loss for {}".
                                  format(config))
