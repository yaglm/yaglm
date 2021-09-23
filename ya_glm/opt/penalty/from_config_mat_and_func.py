from scipy.sparse import identity

from ya_glm.utils import is_str_and_matches
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

    # identify transform
    if isinstance(config, (NoPenalty, Ridge, Lasso, GroupLasso,
                           ExclusiveGroupLasso, NuclearNorm)):
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
                            weights=config.weights)
        # func = get_penalty_func(func_config)

        mat = config.mat

    # fused Lasso
    elif isinstance(config, FusedLasso):
        func_config = Lasso(pen_val=config.pen_val,
                            weights=config.weights)

        # get trend filstering difference matrix
        if is_str_and_matches(config.edgelist, 'chain'):
            mat = get_tf_mat(d=n_features, k=config.order)
        else:
            mat = get_graph_tf_mat(edgelist=config.edgelist,
                                   n_nodes=n_features,
                                   k=config.order)

    return mat, func_config
