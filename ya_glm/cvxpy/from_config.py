import cvxpy as cp

from ya_glm.config.loss import LinReg, L2Reg, Huber, LogReg, Quantile, \
    Poisson, Multinomial

from ya_glm.config.penalty import NoPenalty, Ridge, GeneralizedRidge,\
    Lasso, GroupLasso, MultiTaskLasso, GeneralizedLasso, FusedLasso,\
    OverlappingSum, SeparableSum

from ya_glm.config.constraint import Positive

from ya_glm.cvxpy.glm_loss import lin_reg_loss, log_reg_loss, \
    quantile_reg_loss, l2_reg_loss, huber_reg_loss, poisson_reg_loss, \
    multinomial_loss

from ya_glm.cvxpy.penalty import zero, ridge_penalty, lasso_penalty,\
    gen_ridge_penalty, multi_task_lasso_penalty, group_lasso_penalty, \
    gen_lasso_penalty

from ya_glm.config.base_penalty import get_flavor_info
from ya_glm.opt.from_config.penalty import get_fused_lasso_diff_mat


def get_loss(coef, intercept, X, y, config, sample_weight=None):
    """
    Parameters
    ----------
    coef: cvxpy.Variable
        The initialized cvxpy coefficient variable.

    intercept: None, cvxpy.Variable
        The initialized cvxpy intercept variable or None.

    X, y: array-like
        The data.

    config: LossConfig
        The loss function config.

    sample_weight: None, array-like
        (Optional) Sample weights.

    Output
    ------
    loss_func: cvxpy Expression
        The loss function
    """
    if sample_weight is not None:
        raise NotImplementedError()

    kws = {'coef': coef, 'intercept': intercept,
           'X': X, 'y': y}  # 'sample_weight': sample_weight

    if isinstance(config, LinReg):
        func = lin_reg_loss

    elif isinstance(config, L2Reg):
        func = l2_reg_loss

    elif isinstance(config, Huber):
        func = huber_reg_loss
        kws['knot'] = config.knot

    elif isinstance(config, Poisson):
        func = poisson_reg_loss

    elif isinstance(config, Multinomial):
        func = multinomial_loss

    elif isinstance(config, LogReg):
        func = log_reg_loss

    elif isinstance(config, Quantile):
        func = quantile_reg_loss
        kws['quantile'] = config.quantile

    else:
        raise NotImplementedError("{} not currently available".format(config))

    return func(**kws)


def get_penalty(coef, config):
    """
    Parameters
    ----------
    coef: cvxpy.Variable
        The initialized cvxpy coefficient variable.

    config: PenaltyConfig
        The penalty function config.

    Output
    ------
    penalty_func, pen_val, weights

    penalty_func: cvxpy Expression
        The loss function

    pen_val: cvxpy Parameter, dict
        The penatly value as a cvxpy parameter; this can be modified in place.

    weights: None, cvxpy Parameter, dict
        The weights value as a cvxpy parameter; this can be modified in place.
    """

    flavor_type = get_flavor_info(config)
    if flavor_type == 'non_convex':
        raise NotImplementedError("Cvxpy only works for convex problems!")

    n_features = coef.shape[0]

    kws = {'coef': coef}

    # no penalty
    if isinstance(config, NoPenalty):
        func = zero
        weights = None
        pen_val = None

    # Ridge
    elif isinstance(config, Ridge):
        func = ridge_penalty

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

    # Generalized Ridge
    elif isinstance(config, GeneralizedRidge):
        func = gen_ridge_penalty
        pen_val = cp.Parameter(value=config.pen_val, pos=True)
        weights = None

        kws['mat'] = config.mat
        kws['pen_val'] = pen_val

    # Lasso
    elif isinstance(config, Lasso):
        func = lasso_penalty

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

    # group Lasso
    elif isinstance(config, GroupLasso):

        func = group_lasso_penalty
        pen_val = cp.Parameter(value=config.pen_val, pos=True)

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

        kws['groups'] = config.groups

    # multi task Lasso
    elif isinstance(config, MultiTaskLasso):
        func = multi_task_lasso_penalty
        pen_val = cp.Parameter(value=config.pen_val, pos=True)

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

    # Fused Lasso
    elif isinstance(config, FusedLasso):
        func = gen_lasso_penalty

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

        kws['mat'] = get_fused_lasso_diff_mat(config=config,
                                              n_nodes=n_features)

    # Generalized Lasso
    elif isinstance(config, GeneralizedLasso):
        func = gen_lasso_penalty
        pen_val = cp.Parameter(value=config.pen_val, pos=True)

        # get the pen_val and weights correctly as cp.Parameter
        pen_val, weights = \
            get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
                                             weights=config.weights)

        # add to function's kws
        kws['weights'] = weights
        kws['pen_val'] = pen_val

        kws['mat'] = config.mat

    # Elastic Net
    # elif isinstance(config, ElasticNet):
    # TODO: weights are handled a bit differently
    #     func = elastic_net_penalty
    #     pen_val = cp.Parameter(value=config.pen_val, pos=True)

    #     # get the pen_val and weights correctly as cp.Parameter
    #     pen_val, weights = \
    #         get_pen_val_and_weights_for_prod(pen_val=config.pen_val,
    #                                          weights=config.weights)

    #     # add to function's kws
    #     kws['weights'] = weights
    #     kws['pen_val'] = pen_val

    #     kws['mix_val'] = config.mix_val

    # Overlapping sum
    elif isinstance(config, OverlappingSum):

        penalties, pen_val, weights = \
             zip(*[get_penalty(coef, config=c)
                   for c in config.get_penalties().values()])

        return sum(penalties), pen_val, weights

    # Separable sum
    elif isinstance(config, SeparableSum):
        groups = list(config.get_groups().values())

        penalties, pen_val, weights = \
            zip(*[get_penalty(coef[groups[i]], config=c)
                  for (i, c) in
                  enumerate(config.get_penalties().values())])

        return sum(penalties), pen_val, weights

    else:
        raise NotImplementedError("{} not currently available".format(config))

    return func(**kws), pen_val, weights


def update_pen_val_and_weights(config, pen_val, weights):
    """
    Updates the penatly value or weights cp.Parameter value given a new penalty config object.

    Parameters
    ----------
    config: PenaltyConfig
        The new penalty setting.

    pen_val: cp.Parameter, float
        The cp.Parameter to updated in place.

    weights: cp.Parameter, array-like
        The cp.Parameter to updated in place.
    """

    if isinstance(config, NoPenalty):
        pass

    # Most penalties
    elif isinstance(config, (Ridge, Lasso, GroupLasso, MultiTaskLasso,
                             GeneralizedLasso, FusedLasso)):

        update_weights_and_pen_val_for_prod(pen_val=pen_val,
                                            new_pen_val=config.pen_val,
                                            weights=weights,
                                            new_weights=config.weights)

    # Generalized ridge
    elif isinstance(config, GeneralizedRidge):
        # generalized ridge has no weights
        pen_val.value = config.pen_Val

    # additive penalties
    elif isinstance(config, (SeparableSum, OverlappingSum)):
        configs = config.get_penalties().values()

        for i, cfg in enumerate(configs):
            update_pen_val_and_weights(cfg, pen_val=pen_val[i],
                                       weights=weights[i])

    else:
        raise NotImplementedError("{} not currently available".format(config))


def get_constraints(coef, config=None):
    """
    Parameters
    ----------
    coef: cvxpy.Variable
        The initialized cvxpy coefficient variable.

    config: None, ConstraintConfig
        The constraint config

    Output
    ------
    constraints: None, cvxpy Expression or list
        The constraints applied to coef.
    """

    if config is None:
        return None

    elif isinstance(config, Positive):
        return [coef >= 0]

    else:
        raise NotImplementedError("{} not currently available".format(config))


def get_pen_val_and_weights_for_prod(pen_val, weights=None):
    """
    Cvxpy does not like multiplying cp.Parameters, which are useful for re-using computation. See https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming.

    Safely gets the penalty value and weights for penalties where the penalty value directly mutiplies the weights. Ensures only one of the output items is a cp.Parameter.

    If there are both weights and a pen_val then weights = weights * pen_val and pen_val = 1.

    Parameters
    ----------
    pen_val: float
        The penatly value.

    weights: None, array-like
        The weights

    Output
    ------
    pen_val: float  or cp.Parameter
        The penalty value as either a cp.Parameter (if there are no weights) or a float (if there are weights).

    weights: None, cp.Parameter
        The weigths. None if there are no weights, a cp.Parameter if there are weights.
    """

    if weights is None:
        pen_val = cp.Parameter(value=pen_val, pos=True)

    else:
        weights = cp.Parameter(value=weights * pen_val,
                               shape=weights.shape, pos=True)
        pen_val = 1

    return pen_val, weights


def update_weights_and_pen_val_for_prod(pen_val, new_pen_val,
                                        weights=None,
                                        new_weights=None):
    """
    Updates the cp.Parameter values for the weights and penalty value output by get_pen_val_and_weights_for_prod.
    """
    if new_weights is None:
        pen_val.value = pen_val
    else:
        weights.value = new_pen_val * new_weights
