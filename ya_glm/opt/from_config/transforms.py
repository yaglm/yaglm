import numpy as np
from scipy.linalg import svd
from functools import partial

from ya_glm.linalg_utils import euclid_norm
from ya_glm.trend_filtering import get_tf_mat, get_graph_tf_mat

from ya_glm.config.base_penalty import SeparableSumConfig, ElasticNetConfig
from ya_glm.config.penalty_utils import build_penalty_tree, get_parent_key, \
    extract_penalties, extract_flavors_and_pens, get_enet_sum_name, \
    get_ancestor_keys

from ya_glm.config.penalty import \
    NoPenalty,\
    Ridge,\
    GeneralizedRidge,\
    Lasso,\
    GroupLasso,\
    MultiTaskLasso,\
    ExclusiveGroupLasso,\
    NuclearNorm,\
    FusedLasso,\
    GeneralizedLasso


# TODO: document!!
def get_non_smooth_transforms(config):

    # No transform for smooth penalties
    if isinstance(config, (NoPenalty, Ridge, GeneralizedRidge)) \
            or config is None:
        return None

    # Lasso
    elif isinstance(config, Lasso):
        return entrywise_abs_transform

    # Group Lasso
    elif isinstance(config, GroupLasso):
        return partial(group_transform, groups=config.groups)

    # Exclusive group lasso
    elif isinstance(config, ExclusiveGroupLasso):
        raise NotImplementedError("TODO: need to think through "
                                  "what this looks lke")

    # Multi-task lasso
    elif isinstance(config, MultiTaskLasso):
        return multi_task_lasso_transform

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
    # TODO: cache the diff mat!

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


def get_group_func(func, grp_idxs):
    if func is None:
        return None

    def group_func(coef):
        return func(coef[grp_idxs])

    return group_func


def get_flavored_transforms(penalty, kind):
    """
    Gets the tranformation functions for all flavored penalties.

    Parameters
    ----------
    penalty: PenaltyConfig, PenaltyTuner
        The penalty configs/tuner whose flavored penalties we want.

    kind: str
        Which kind of flavor we want; ['adaptive', 'non_convex']

    Output
    ------
    transforms, flavors, flavor_keys

    transforms: list of callable(coef)

    flavors: list of FlavorConfigs

    flavor_keys: list of str
    """

    tree = build_penalty_tree(penalty)
    penalties, penalty_keys = extract_penalties(tree)

    #########################
    # get requested flavors #
    #########################

    flavors, flavor_keys, parent_pens = \
        extract_flavors_and_pens(penalties=penalties,
                                 penalty_keys=penalty_keys,
                                 restrict=kind)

    ############################################
    # get transforms for each flavored penalty #
    ############################################

    # pull out the transform for each flavor
    transforms = []

    for flav_key, parent_pen in zip(flavor_keys, parent_pens):

        # for elastic net we need to figure out which term
        # in the sum this flavor is
        if isinstance(parent_pen, ElasticNetConfig):

            # get the config for the enet summand corresponding
            # to this flavor config
            sum_name = get_enet_sum_name(flav_key)
            sum_config = parent_pen.get_sum_configs(sum_name)

            transf = get_non_smooth_transforms(sum_config)
        else:
            transf = get_non_smooth_transforms(parent_pen)

        transforms.append(transf)

    ###########################################
    # add group subsetting to transformations #
    ###########################################

    # get keys of penalties that are Children of SeparableSum penalties
    # and the corresponding groups
    CoSS_keys, groups = extract_grouping_info(penalties=penalties,
                                              keys=penalty_keys)

    # if there are no group subsetting functions then we are done!
    if len(groups) == 0:
        return transforms, flavors, flavor_keys

    # map keys to groups
    group_map = {k: groups[idx] for idx, k in enumerate(CoSS_keys)}

    # get the ancestor keys among the CoSS_keys for each flavored key
    flav_groups = {}
    for flav_key in flavor_keys:
        # get ancestors
        ancestors = get_ancestor_keys(key=flav_key,
                                      candidates=CoSS_keys)

        # ancestor groups starting with the oldest ancestors
        flav_groups[flav_key] = [group_map[a] for a in ancestors]

    # wrap the transforms in a subsetting function
    subsetted_transforms = []
    for i, flav_key in enumerate(flavor_keys):
        new_transf = wrap_group_subsetting(func=transforms[i],
                                           groups=flav_groups[flav_key])

        subsetted_transforms.append(new_transf)

    return subsetted_transforms, flavors, flavor_keys


def wrap_group_subsetting(func, groups):
    """
    Parameters
    ----------
    func: callable(coef) -> array-like
        The original function to wrap.

    groups: list of lists
        The lists group indices sorted in order the subsetting should be applied.

    Output
    -----
    subsetted_func: callable(coef) -> array-like
        The function applied to the subsetted coefficient.
    """
    def subsetted_func(coef):
        for i, group_idxs in enumerate(groups):
            if i == 0:
                coef_subsetted = coef[group_idxs]
            else:
                coef_subsetted = coef_subsetted[group_idxs]

        return func(coef_subsetted)

    return subsetted_func


def extract_grouping_info(penalties, keys):
    """
    Parameters
    ----------
    TODO:
    Output
    ------
    keys, groups

    keys: list of str
        Keys to penalties whose parents are SeparableSum penalties.

    groups: list of lists
        The groups corresponding to each child.
    """

    # pull out all the separable sum penalties that have groups
    sep_sum_keys = []
    sep_sum_pens = []
    for idx, pen in enumerate(penalties):
        if isinstance(pen, SeparableSumConfig):
            pen_key = keys[idx]
            sep_sum_keys.append(pen_key)

            sep_sum_pens.append(pen)

    sep_sum_map = {k: sep_sum_pens[idx] for idx, k in enumerate(sep_sum_keys)}

    # pull out all the penalties whose are children of a separable sum
    # and the corresponding groups
    groups = []
    children_keys = []
    for idx, key in enumerate(keys):

        # if the parent is a separable sum pull out the groups
        parent_key = get_parent_key(key)
        if parent_key in sep_sum_keys:

            # this child's parent is a sep sum
            children_keys.append(key)

            # pull out the groups at this stage
            this_name = key.split('__')[-1]
            parent_pen = sep_sum_map[parent_key]
            this_group = parent_pen.get_groups()[this_name]
            groups.append(this_group)

    return children_keys, groups
