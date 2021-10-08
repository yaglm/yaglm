from yaglm.opt.glm_loss.linear_regression import LeastSquares, \
    LeastSquaresMulti
from yaglm.opt.glm_loss.L2_regression import L2Loss, L2LossMulti
from yaglm.opt.glm_loss.huber_regression import Huber, HuberMulti
from yaglm.opt.glm_loss.poisson_regression import Poisson, PoissonMulti
from yaglm.opt.glm_loss.logistic_regression import Logistic
from yaglm.opt.glm_loss.multinomial import MultinomialLoss
from yaglm.opt.glm_loss.quantile_regression import Quantile, QuantileMulti


def get_glm_input_loss(config, y, sample_weight=None):

    is_mr = y.ndim == 2 and y.shape[1] > 1
    kws = {'y': y, 'sample_weight': sample_weight}

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

    # L2 regression
    elif config.name == 'l2':
        if is_mr:
            return L2LossMulti(**kws)
        else:
            return L2Loss(**kws)

    else:
        raise NotImplementedError("Could not find input loss for {}".
                                  format(config))
