from ya_glm.config.penalty import NoPenalty
from ya_glm.config.penalty import Ridge as RidgeConfig
from ya_glm.config.penalty import GeneralizedRidge as GeneralizedRidgeConfig

from ya_glm.config.penalty import Lasso as LassoConfig
from ya_glm.config.penalty import GroupLasso as GroupLassoConfig
from ya_glm.config.penalty import \
     ExclusiveGroupLasso as ExclusiveGroupLassoConfig

from ya_glm.config.penalty import MultiTaskLasso as MultiTaskLassoConfig
from ya_glm.config.penalty import NuclearNorm as NuclearNormConfig

from ya_glm.opt.base import Zero
from ya_glm.opt.penalty.convex import Ridge, GeneralizedRidge,\
     Lasso, GroupLasso, ExclusiveGroupLasso, \
     MultiTaskLasso, NuclearNorm

from ya_glm.opt.penalty.nonconvex import get_nonconvex_func
from ya_glm.opt.penalty.composite_structured import CompositeGroup, \
    CompositeMultiTaskLasso, CompositeNuclearNorm

from ya_glm.opt.penalty.with_intercept import MatWithIntercept, WithIntercept


def get_penalty_func(config):
    """

    Parameters
    ----------
    config: PenaltyConfig
        The penalty congig object.
    """

    # no penalty!
    if isinstance(config, NoPenalty):
        return Zero()

    # Ridge penalty
    if isinstance(config, RidgeConfig):
        return Ridge(pen_val=config.pen_val, weights=config.weights)

    # Generalized ridge penalty
    if isinstance(config, GeneralizedRidgeConfig):
        return GeneralizedRidge(pen_val=config.pen_val,
                                mat=config.mat)

    # Entrywise penalties e.g. lasso, SCAD, etc
    elif isinstance(config, LassoConfig):
        if config.flavor is None or config.flavor.name == 'adaptive':
            return Lasso(pen_val=config.pen_val, weights=config.weights)

        else:
            # non-convex
            return _get_nonconvex(config)

    # Group penalties e.g. group lasso, group scad etc
    elif isinstance(config, GroupLassoConfig):
        if config.flavor is None or config.flavor.name == 'adaptive':
            return GroupLasso(groups=config.groups,
                              pen_val=config.pen_val,
                              weights=config.weights)

        else:
            # get non-convex func
            nc_func = _get_nonconvex(config)

            return CompositeGroup(groups=config.groups,
                                  func=nc_func)

    # Exclusive group lasso
    elif isinstance(config, ExclusiveGroupLassoConfig):
        return ExclusiveGroupLasso(groups=config.groups,
                                   pen_val=config.pen_val)

    # Multitask e.g. multi-task lasso, multi-task scad etc
    elif isinstance(config, MultiTaskLassoConfig):
        if config.flavor is None or config.flavor.name == 'adaptive':
            return MultiTaskLasso(pen_val=config.pen_val,
                                  weights=config.weights)

        else:
            # get non-convex func
            nc_func = _get_nonconvex(config)
            return CompositeMultiTaskLasso(func=nc_func)

    # Nuclear norm, adaptive nuclear norm or non-convex nuclear norm
    elif isinstance(config, NuclearNormConfig):
        if config.flavor is None or config.flavor.name == 'adaptive':
            return NuclearNorm(pen_val=config.pen_val,
                               weights=config.weights)

        else:
            # get non-convex func
            nc_func = _get_nonconvex(config)
            return CompositeNuclearNorm(func=nc_func)

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "ya_glm.opt.penalty".
                                  format(config))


def _get_nonconvex(config):
    return get_nonconvex_func(name=config.flavor.pen_func,
                              pen_val=config.pen_val,
                              second_param=config.flavor.
                              second_param_val)


def wrap_intercept(func, fit_intercept, is_mr):
    """

    Parameters
    ----------
    config: PenaltyConfig
        The penalty congig object.
    """

    if fit_intercept:
        if is_mr:
            return MatWithIntercept(func=func)

        else:
            return WithIntercept(func=func)

    else:
        return func
