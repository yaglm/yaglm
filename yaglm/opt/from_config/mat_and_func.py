from scipy.sparse import identity, csr_matrix, bmat, eye
import numpy as np

from ya_glm.opt.from_config.penalty import get_fused_lasso_diff_mat

from ya_glm.config.penalty import \
    NoPenalty,\
    Ridge,\
    GeneralizedRidge,\
    Lasso,\
    GroupLasso,\
    ExclusiveGroupLasso,\
    MultiTaskLasso,\
    NuclearNorm,\
    FusedLasso,\
    GeneralizedLasso,\
    ElasticNet,\
    GroupElasticNet,\
    MultiTaskElasticNet,\
    SparseGroupLasso,\
    SeparableSum, \
    OverlappingSum  # InifmalSum,


def get_mat_and_func(config, n_features):
    """
    Gets the penalty function and linear transformation for the augmented ADMM algorithm of (Zhu, 2017).

    We can write decompose each penalty as
    p_{pen_val}(coef; weights) = q_{pen_val}(mat @coef; weights)

    for some other penalty q and matrix mat. For penalties like the lasso p=q and mat = identity. For the genrealized lasso p=lasso and mat is the generalized Lasso transform.

    Parameters
    ----------
    config: PenaltyConfig
        The penalty config.

    n_features: int
        Number of features.

    Output
    ------
    mat, func_config

    mat: array-like, (n_transf, n_features)
        The linear transformation matrix, mat. This is often a sparse matrix.

    func_config: PenaltyConfig
        The config for the penalty function, q, applied to the transformed coefficient.

    References
    ----------
    Zhu, Y., 2017. An augmented ADMM algorithm with application to the generalized lasso problem. Journal of Computational and Graphical Statistics, 26(1), pp.195-204.
    """

    # TODO: perhaps for NoPenalty return zero
    if isinstance(config, NoPenalty) or config is None:
        func_config = config
        mat = csr_matrix((n_features, n_features))

    # identify transform
    elif isinstance(config, (Ridge, Lasso,
                             GroupLasso,
                             ExclusiveGroupLasso,
                             MultiTaskLasso, NuclearNorm,
                             ElasticNet,
                             GroupElasticNet,
                             MultiTaskElasticNet,
                             SparseGroupLasso,
                             SeparableSum)):

        func_config = config
        mat = identity(n_features)

    # generalized Ridge
    elif isinstance(config, GeneralizedRidge):

        # TODO: weights? I don't think we need them
        func_config = Ridge(pen_val=config.pen_val)
        mat = config.mat

    # generalized Lasso
    elif isinstance(config, GeneralizedLasso):

        # TODO: weights? I don't think we need them
        func_config = Lasso(pen_val=config.pen_val,
                            weights=config.weights,
                            flavor=config.flavor)

        mat = config.mat

    # fused Lasso
    elif isinstance(config, FusedLasso):
        func_config = Lasso(pen_val=config.pen_val,
                            weights=config.weights,
                            flavor=config.flavor)

        # get trend filtering difference matrix
        mat = get_fused_lasso_diff_mat(config, n_nodes=n_features)

    # overlaping sum
    elif isinstance(config, OverlappingSum):

        # Our ADMM formulation converts an overlapping sum to a separable sum
        # by creating a new coefficient for each penalty by stacking identities

        # pull out penalties
        penalties = config.get_penalties()

        # stack identities on top of each other
        n_funcs = len(penalties)
        mat = bmat([[eye(n_features)] for b in range(n_funcs)])

        # group indices are consecutive blocks
        groups = get_consecutive_groups(n_features=n_features,
                                        n_groups=len(penalties))

        groups = {name: groups[idx]  # convert to a dict
                  for (idx, name) in enumerate(penalties.keys())}

        # convert overlapping sum to separable sum
        func_config = SeparableSum(groups=groups, **penalties)

    else:
        raise NotImplementedError("Not available for {}".format(config))

    return mat, func_config


def get_consecutive_groups(n_features, n_groups):
    """
    Gets groups of consecutive indices.

    Parameters
    ----------
    n_features: int
        Number of features (length of each group).

    n_groups: int
        Number of groups.

    Output
    ------
    groups: list of array-like
        The group indices.
    """
    groups = []
    left, right = 0, 0
    for _ in range(n_groups):
        right += n_features
        groups.append(np.arange(left, right))
        left += n_features
    return groups
