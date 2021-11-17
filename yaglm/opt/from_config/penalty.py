from yaglm.config.penalty import NoPenalty
from yaglm.config.penalty import Ridge as RidgeConfig
from yaglm.config.penalty import GeneralizedRidge as GeneralizedRidgeConfig
from yaglm.config.penalty import Lasso as LassoConfig
from yaglm.config.penalty import GroupLasso as GroupLassoConfig
from yaglm.config.penalty import \
     ExclusiveGroupLasso as ExclusiveGroupLassoConfig
from yaglm.config.penalty import MultiTaskLasso as MultiTaskLassoConfig
from yaglm.config.penalty import NuclearNorm as NuclearNormConfig
from yaglm.config.penalty import FusedLasso as FusedLassoConfig
from yaglm.config.penalty import GeneralizedLasso as GeneralizedLassoConfig
from yaglm.config.penalty import ElasticNet as ElasticNetConfig
from yaglm.config.penalty import GroupElasticNet as GroupElasticNetConfig
from yaglm.config.penalty import MultiTaskElasticNet as \
    MultiTaskElasticNetConfig
from yaglm.config.penalty import SparseGroupLasso as SparseGroupLassoConfig
from yaglm.config.penalty import SeparableSum as SeparableSumConfig
# from yaglm.config.penalty import InifmalSum as InifmalSumConfig
from yaglm.config.penalty import OverlappingSum as OverlappingSumConfig

from yaglm.config.base_penalty import WithFlavorPenSeqConfig

from yaglm.opt.base import Sum
from yaglm.opt.BlockSeparable import BlockSeparable
from yaglm.opt.penalty.convex import Ridge, GeneralizedRidge,\
     Lasso, GroupLasso, ExclusiveGroupLasso, \
     MultiTaskLasso, NuclearNorm, GeneralizedLasso, \
     ElasticNet, GroupElasticNet, MultiTaskElasticNet, SparseGroupLasso
from yaglm.opt.penalty.nonconvex import get_nonconvex_func
from yaglm.opt.penalty.composite_structured import CompositeGroup, \
    CompositeMultiTaskLasso, CompositeNuclearNorm, CompositeGeneralizedLasso
from yaglm.opt.penalty.utils import MatWithIntercept, WithIntercept

from yaglm.utils import is_str_and_matches
from yaglm.trend_filtering import get_tf_mat, get_graph_tf_mat
from yaglm.config.penalty_utils import get_flavor_kind


def get_penalty_func(config, n_features=None, n_responses=None):
    """
    Gets a penalty function from a PenaltyConfig object.

    Parameters
    ----------
    config: PenaltyConfig
        The penalty config object.

    n_features: None, int
        (Optional) Number of features the penalty will be applied to. This is only needed for the fused Lasso and the

    Output
    ------
    func: yaglm.opt.base.Func
        The penalty function
    """

    flavor_kind = get_flavor_kind(config)

    # no penalty!
    if config is None or isinstance(config, NoPenalty):
        return None

    # Ridge penalty
    elif isinstance(config, RidgeConfig):
        return Ridge(pen_val=config.pen_val, weights=config.weights)

    # Generalized ridge penalty
    elif isinstance(config, GeneralizedRidgeConfig):
        return GeneralizedRidge(pen_val=config.pen_val,
                                mat=config.mat)

    # Entrywise penalties e.g. lasso, SCAD, etc
    elif isinstance(config, LassoConfig):
        if flavor_kind == 'non_convex':
            return get_outer_nonconvex_func(config)
        else:

            return Lasso(pen_val=config.pen_val, weights=config.weights)

    # Group penalties e.g. group lasso, group scad etc
    elif isinstance(config, GroupLassoConfig):
        if flavor_kind == 'non_convex':
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
        if flavor_kind is not None:
            raise NotImplementedError()

        return ExclusiveGroupLasso(groups=config.groups,
                                   pen_val=config.pen_val)

    # Multitask e.g. multi-task lasso, multi-task scad etc
    elif isinstance(config, MultiTaskLassoConfig):

        if flavor_kind == 'non_convex':
            # get non-convex func
            nc_func = get_outer_nonconvex_func(config)
            return CompositeMultiTaskLasso(func=nc_func)
        else:
            return MultiTaskLasso(pen_val=config.pen_val,
                                  weights=config.weights)

    # Nuclear norm, adaptive nuclear norm or non-convex nuclear norm
    elif isinstance(config, NuclearNormConfig):
        if flavor_kind == 'non_convex':
            # get non-convex func
            nc_func = get_outer_nonconvex_func(config)
            return CompositeNuclearNorm(func=nc_func)

        else:
            return NuclearNorm(pen_val=config.pen_val,
                               weights=config.weights)

    # Generalized and fused lasso
    elif isinstance(config, (FusedLassoConfig, GeneralizedLassoConfig)):
        # TODO: perhaps add separate TV-1

        if isinstance(config, FusedLassoConfig):
            mat = get_fused_lasso_diff_mat(config=config, n_nodes=n_features)
        else:
            mat = config.mat

        if flavor_kind == 'non_convex':
            nc_func = get_outer_nonconvex_func(config)
            return CompositeGeneralizedLasso(func=nc_func, mat=mat)
        else:
            return GeneralizedLasso(pen_val=config.pen_val,
                                    mat=mat,
                                    weights=config.weights)

    # Elastic Net
    elif isinstance(config, ElasticNetConfig):

        if flavor_kind == 'non_convex':
            # sums the two individual penalties
            return get_enet_sum(config)

        else:
            return ElasticNet(pen_val=config.pen_val,
                              mix_val=config.mix_val,
                              lasso_weights=config.lasso_weights,
                              ridge_weights=config.ridge_weights
                              )

    # Group Elastic net
    elif isinstance(config, GroupElasticNetConfig):
        if flavor_kind == 'non_convex':
            # sums the two individual penalties
            return get_enet_sum(config)

        else:
            return GroupElasticNet(groups=config.groups,
                                   pen_val=config.pen_val,
                                   mix_val=config.mix_val,
                                   lasso_weights=config.lasso_weights,
                                   ridge_weights=config.ridge_weights
                                   )

    # Multi-task elastic net
    elif isinstance(config, MultiTaskElasticNetConfig):
        if flavor_kind == 'non_convex':
            # sums the two individual penalties
            return get_enet_sum(config)

        else:
            return MultiTaskElasticNet(pen_val=config.pen_val,
                                       mix_val=config.mix_val,
                                       lasso_weights=config.lasso_weights,
                                       ridge_weights=config.ridge_weights
                                       )

    # Sparse group lasso
    elif isinstance(config, SparseGroupLassoConfig):

        if flavor_kind == 'non_convex':
            # sums the two individual penalties
            return get_enet_sum(config)

        else:
            return SparseGroupLasso(groups=config.groups,
                                    pen_val=config.pen_val,
                                    mix_val=config.mix_val,
                                    sparse_weights=config.sparse_weights,
                                    group_weights=config.group_weights)

    # Overlapping sum
    elif isinstance(config, OverlappingSumConfig):
        funcs = [get_penalty_func(config=c,
                                  n_features=n_features,
                                  n_responses=n_responses)
                 for c in config.get_penalties().values()]
        return Sum(funcs=funcs)

    # Separable sum
    elif isinstance(config, SeparableSumConfig):
        funcs = [get_penalty_func(config=c,
                                  n_features=n_features,
                                  n_responses=n_responses)
                 for c in config.get_penalties().values()]

        groups = [grp_idxs for grp_idxs in config.get_groups().values()]

        return BlockSeparable(funcs=funcs, groups=groups)

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "yaglm.opt.penalty".
                                  format(config))


def get_enet_sum(config):
    # TODO: document
    # This just sums the two elastic net terms
    # TODO: are they any nice combined proxs for sums of non-convex?
    # for non-convex versions of the elastic net I'm not sure we have
    # the nice prox decomposition formulas for things like the sparse
    # group lasso. If we do, then we can do something smarter than this

    # pull out penalties for each term in the sum
    sum_configs = config.get_sum_configs()
    pen1 = get_penalty_func(sum_configs[0])
    pen2 = get_penalty_func(sum_configs[1])
    return Sum([pen1, pen2])


def get_outer_nonconvex_func(config):
    """
    Returns the non-convex function used in a non-convex penalty. If the overall penalty is a composition, p(coef) = non-convex(t(coef)), this returns the final non-convex function.

    Parameters
    ----------
    config: PenaltyConfig
        A non-convex penalty config.

    Output
    ------
    func: yaglm.opt.base.func
        The non-convex function.
    """

    if isinstance(config, WithFlavorPenSeqConfig):
        if config.flavor is None:
            return None

        return get_nonconvex_func(name=config.flavor.pen_func,
                                  pen_val=config.pen_val,
                                  second_param=config.flavor.
                                  second_param_val)

    else:
        raise RuntimeError("Unable to get outer non-convex function")


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
