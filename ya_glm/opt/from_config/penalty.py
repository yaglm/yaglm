from ya_glm.config.penalty import NoPenalty
from ya_glm.config.penalty import Ridge as RidgeConfig
from ya_glm.config.penalty import GeneralizedRidge as GeneralizedRidgeConfig

from ya_glm.config.penalty import Lasso as LassoConfig
from ya_glm.config.penalty import GroupLasso as GroupLassoConfig
from ya_glm.config.penalty import \
     ExclusiveGroupLasso as ExclusiveGroupLassoConfig

from ya_glm.config.penalty import MultiTaskLasso as MultiTaskLassoConfig
from ya_glm.config.penalty import NuclearNorm as NuclearNormConfig

from ya_glm.config.penalty import FusedLasso as FusedLassoConfig
from ya_glm.config.penalty import GeneralizedLasso as GeneralizedLassoConfig

from ya_glm.config.penalty import ElasticNet as ElasticNetConfig
from ya_glm.config.penalty import GroupElasticNet as GroupElasticNetConfig
from ya_glm.config.penalty import MultiTaskElasticNet as \
    MultiTaskElasticNetConfig
from ya_glm.config.penalty import SparseGroupLasso as SparseGroupLassoConfig

# from ya_glm.config.penalty import SeparableSum as SeparableSumConfig
# from ya_glm.config.penalty import InifmalSum as InifmalSumConfig
from ya_glm.config.penalty import OverlappingSum as OverlappingSumConfig

from ya_glm.opt.base import Zero, Sum
from ya_glm.opt.penalty.convex import Ridge, GeneralizedRidge,\
     Lasso, GroupLasso, ExclusiveGroupLasso, \
     MultiTaskLasso, NuclearNorm, GeneralizedLasso, \
     ElasticNet, GroupElasticNet, MultiTaskElasticNet, SparseGroupLasso

from ya_glm.opt.penalty.nonconvex import get_nonconvex_func
from ya_glm.opt.penalty.composite_structured import CompositeGroup, \
    CompositeMultiTaskLasso, CompositeNuclearNorm, CompositeGeneralizedLasso

from ya_glm.opt.penalty.utils import MatWithIntercept, WithIntercept
from ya_glm.utils import is_str_and_matches
from ya_glm.trend_filtering import get_tf_mat, get_graph_tf_mat
from ya_glm.config.base_penalty import get_flavor_info


def get_penalty_func(config, n_features=None):
    """
    Gets a penalty function from a PenaltyConfig object.

    Parameters
    ----------
    config: PenaltyConfig
        The penalty congig object.

    n_features: None, int
        (Optional) Number of features the penalty will be applied to. This is only needed for the fused Lasso.

    Output
    ------
    func: ya_glm.opt.base.Func
        The penalty function
    """

    # no penalty!
    if config is None or isinstance(config, NoPenalty):
        return Zero()

    flavor_type = get_flavor_info(config)

    # Ridge penalty
    if isinstance(config, RidgeConfig):
        return Ridge(pen_val=config.pen_val, weights=config.weights)

    # Generalized ridge penalty
    elif isinstance(config, GeneralizedRidgeConfig):
        return GeneralizedRidge(pen_val=config.pen_val,
                                mat=config.mat)

    # Entrywise penalties e.g. lasso, SCAD, etc
    elif isinstance(config, LassoConfig):
        if flavor_type == 'non_convex':
            return get_outer_nonconvex_func(config)
        else:

            return Lasso(pen_val=config.pen_val, weights=config.weights)

    # Group penalties e.g. group lasso, group scad etc
    elif isinstance(config, GroupLassoConfig):
        if flavor_type == 'non_convex':
            # get non-convex func
            nc_func = get_outer_nonconvex_func(config)
            return CompositeGroup(groups=config.groups,
                                  func=nc_func)
        else:
            return GroupLasso(groups=config.groups,
                              pen_val=config.pen_val,
                              weights=config.weights)

    # Exclusive group lasso
    elif isinstance(config, ExclusiveGroupLassoConfig):
        if flavor_type is not None:
            raise NotImplementedError()

        return ExclusiveGroupLasso(groups=config.groups,
                                   pen_val=config.pen_val)

    # Multitask e.g. multi-task lasso, multi-task scad etc
    elif isinstance(config, MultiTaskLassoConfig):

        if flavor_type == 'non_convex':
            # get non-convex func
            nc_func = get_outer_nonconvex_func(config)
            return CompositeMultiTaskLasso(func=nc_func)
        else:
            return MultiTaskLasso(pen_val=config.pen_val,
                                  weights=config.weights)

    # Nuclear norm, adaptive nuclear norm or non-convex nuclear norm
    elif isinstance(config, NuclearNormConfig):
        if flavor_type == 'non_convex':
            # get non-convex func
            nc_func = get_outer_nonconvex_func(config)
            return CompositeNuclearNorm(func=nc_func)

        else:
            return NuclearNorm(pen_val=config.pen_val,
                               weights=config.weights)

    # generalized and fused lasso
    elif isinstance(config, (FusedLassoConfig, GeneralizedLassoConfig)):
        # TODO: perhaps add separate TV-1

        if isinstance(config, FusedLassoConfig):
            mat = get_fused_lasso_diff_mat(config=config, n_nodes=n_features)
        else:
            mat = config.mat

        if flavor_type == 'non_convex':
            nc_func = get_outer_nonconvex_func(config)
            return CompositeGeneralizedLasso(func=nc_func, mat=mat)
        else:
            return GeneralizedLasso(pen_val=config.pen_val,
                                    mat=mat,
                                    weights=config.weights)

    # Elastic Net
    elif isinstance(config, ElasticNetConfig):

        if flavor_type == 'non_convex':
            raise NotImplementedError("TODO: add")

        else:

            return ElasticNet(pen_val=config.pen_val,
                              mix_val=config.mix_val,
                              lasso_weights=config.weights
                              )

    # Group Elastic net
    elif isinstance(config, GroupElasticNetConfig):
        if flavor_type == 'non_convex':
            raise NotImplementedError("TODO: add")

        else:
            return GroupElasticNet(groups=config.groups,
                                   pen_val=config.pen_val,
                                   mix_val=config.mix_val,
                                   lasso_weights=config.weights
                                   )

    # Multi-task elastic net
    elif isinstance(config, MultiTaskElasticNetConfig):
        if flavor_type == 'non_convex':
            raise NotImplementedError("TODO: add")

        else:
            return MultiTaskElasticNet(pen_val=config.pen_val,
                                       mix_val=config.mix_val,
                                       lasso_weights=config.weights
                                       )

    # Sparse group lasso
    elif isinstance(config, SparseGroupLassoConfig):

        if flavor_type == 'non_convex':
            raise NotImplementedError("TODO")

        else:
            return SparseGroupLasso(groups=config.groups,
                                    pen_val=config.pen_val,
                                    mix_val=config.mix_val,
                                    sparse_weights=config.sparse_weights,
                                    group_weights=config.group_weights)

    # Overlapping sum
    elif isinstance(config, OverlappingSumConfig):
        funcs = [get_penalty_func(c, n_features)
                 for c in config.get_penalties().values()]
        return Sum(funcs=funcs)

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "ya_glm.opt.penalty".
                                  format(config))


def get_outer_nonconvex_func(config):
    """
    Returns the non-convex function used in a non-convex penalty. If the overall penalty is a composition, p(coef) = non-convex(t(coef)), this returns the final non-convex function.

    Parameters
    ----------
    config: PenaltyConfig
        A non-convex penalty config.

    Output
    ------
    func: ya_glm.opt.base.func
        The non-convex function.
    """
    if get_flavor_info(config) != 'non_convex':
        raise ValueError("Penalty is not non_convex")

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


def get_fused_lasso_diff_mat(config, n_nodes):
    """
    Returns the generalized lasso difference matrix for the fused lasso.

    Parameters
    ----------
    config: FusedLasso
        The fused lasso config.

    n_nodes: int
        The number of nodes in the graph.

    Output
    ------
    mat: array-like, (n_rows, n_nodes)
        The difference matrix as a sparse matrix.
    """

    assert isinstance(config, FusedLassoConfig)

    if is_str_and_matches(config.edgelist, 'chain'):
        return get_tf_mat(d=n_nodes, k=config.order)
    else:
        return get_graph_tf_mat(edgelist=config.edgelist,
                                n_nodes=n_nodes,
                                k=config.order)
