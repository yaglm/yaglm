import numpy as np
from scipy.linalg import svd
from functools import partial

from ya_glm.linalg_utils import euclid_norm
from ya_glm.trend_filtering import get_tf_mat, get_graph_tf_mat

from ya_glm.config.penalty import \
    NoPenalty,\
    Ridge,\
    GeneralizedRidge,\
    Lasso,\
    GroupLasso,\
    ExclusiveGroupLasso,\
    NuclearNorm,\
    FusedLasso,\
    GeneralizedLasso


def get_transform(config):

    # No transform for smooth penalties
    if isinstance(config, (NoPenalty, Ridge, GeneralizedRidge)):
        return None

    # Lasso
    elif isinstance(config, Lasso):
        return entrywise_abs_transform

    # Group Lasso
    elif isinstance(config, GroupLasso):
        return partial(group_transform, groups=config.groups)

    # Exclusive group lasso
    elif isinstance(config, ExclusiveGroupLasso):
        raise NotImplementedError("TODO: need to think through"
                                  "what this looks lke")

    # Nuclear norm
    elif isinstance(config, NuclearNorm):
        return sval_transform

    # Fused Lasso
    elif isinstance(config, FusedLasso):
        return partial(fused_lasso_transform,
                       edgelist=config.edgelist,
                       order=config.order)

    # Generalized Lasso
    elif isinstance(config, GeneralizedLasso):
        return partial(generalized_lasso_transform, config=config.mat)

    else:
        raise NotImplementedError("Could not find transform for {}".
                                  format(config))


def entrywise_abs_transform(coef):
    return abs(coef.reshape(-1))


def multi_task_lasso_transform(coef):
    return np.array([euclid_norm(coef[r, :])
                     for r in range(coef.shape[0])])


def sval_transform(coef):
    return svd(coef)[1]


def group_transform(coef, groups):
    group_norms = []
    for g, grp_idxs in enumerate(groups):
        norm = euclid_norm(coef[grp_idxs])
        norm /= np.sqrt(len(grp_idxs))
        group_norms.append(norm)

    return np.array(group_norms)


def fused_lasso_transform(coef, edgelist, order):

    if coef.ndim == 2:
        raise NotImplementedError("TODO add this case")

    # first order things are easy to compute
    if order == 1:

        if edgelist == 'chain' and order == 1:
            return abs(np.diff(coef))

        else:
            return np.array([abs(coef[edge[1]] - coef[edge[0]])
                             for edge in edgelist])

    else:
        n_nodes = coef.shape[0]

        # get the differencing matrix
        if edgelist == 'chain':
            diff_mat = get_tf_mat(d=n_nodes, k=order)
        else:
            diff_mat = get_graph_tf_mat(edgelist=edgelist, n_nodes=n_nodes,
                                        k=order)

        return abs(diff_mat @ coef).reshape(-1)


def generalized_lasso_transform(coef, mat):
    if mat is None:
        return entrywise_abs_transform(coef)

    return abs(mat @ coef).reshape(-1)
