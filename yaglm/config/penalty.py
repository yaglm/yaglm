from yaglm.config.base_penalty import PenaltyConfig, WithPenSeqConfig, \
    WithFlavorPenSeqConfig, ElasticNetConfig, \
    SeparableSumConfig, InfimalSumConfig, OverlappingSumConfig

from yaglm.pen_max.ridge import get_ridge_pen_max
from yaglm.pen_max.lasso import get_lasso_pen_max

from yaglm.autoassign import autoassign


class NoPenalty(PenaltyConfig):
    """
    Represents no penalty.
    """
    pass


class Ridge(WithPenSeqConfig):
    """
    Ridge penalty with optional weights.

    pen_val * 0.5 * ||coef||_2^2
    pen_val * 0.5 * sum_j weighs_j coef_j^2

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    weights: None, array-like
        (Optional) Weights for each term in the penalty.
    """
    @autoassign
    def __init__(self, pen_val=1, weights=None, targ_ubd=1): pass

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None, init_data=None):

        return get_ridge_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight,
                                 targ_ubd=self.targ_ubd,
                                 norm_by_dim=True)


class GeneralizedRidge(WithPenSeqConfig):
    """
    Generalized ridge penalty with a matrix transform i.e.

    pen_val * 0.5 * ||mat @ coef||_2^2

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    mat: None, array-like (K, n_features)
        The matrix transform.
    """

    @autoassign
    def __init__(self, pen_val=1, mat=None): pass


class Lasso(WithFlavorPenSeqConfig):
    """
    Entrywise non-smooth penalty e.g. a lasso, adapative lasso or entrywise SCAD.

    pen_val * ||coef||_1
    pen_val * sum_j weighs_j |coef_j|
    sum_j non-convex_{pen_val} (coef_j)

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    weights: None, array-like
        (Optional) Weights for each term in the penalty.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.
    """
    @autoassign
    def __init__(self, pen_val=1, weights=None, flavor=None): pass

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


# TODO: add default weights
# TODO: add default to infimal option for overlapping
class GroupLasso(WithFlavorPenSeqConfig):
    """
    Group penalty e.g. group lasso, adaptive group lasso, or group non-convex.

    pen_val * sum_{g in groups} weights_g ||coef_g||_2
    sum_{g in groups} non-convex_{pen_val}(||coef_g||_2)

    Parameters
    ----------
    groups: list, None
        Indices of the groups. If None, then all features are put in a single group.

    pen_val: float
        The penalty parameter value.

    weights: None, array-like
        (Optional) Weights for each term in the penalty.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.

    References
    ----------
    Yuan, M. and Lin, Y., 2006. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 68(1), pp.49-67.

    Breheny, P. and Huang, J., 2015. Group descent algorithms for nonconvex penalized linear and logistic regression models with grouped predictors. Statistics and computing, 25(2), pp.173-187.
    """
    @autoassign
    def __init__(self, groups=None, pen_val=1, weights=None, flavor=None): pass

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 groups=self.groups,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


# TODO: add flavor
# TODO: add weights -- should we have both entrywise and group?
# perhaps default gives group lasso weights
class ExclusiveGroupLasso(WithPenSeqConfig):
    """
    Exclusive group lasso.

    pen_val * sum_{g in groups} ||coef_g||_1^2

    Parameters
    ----------
    groups: list, None
        Indices of the groups. If None, then all features are put in a single group.

    pen_val: float
        The penalty parameter value.

    References
    ----------
    Campbell, F. and Allen, G.I., 2017. Within group variable selection through the exclusive lasso. Electronic Journal of Statistics, 11(2), pp.4220-4257.

    Zhou, Y., Jin, R. and Hoi, S.C.H., 2010, March. Exclusive lasso for multi-task feature selection. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 988-995). JMLR Workshop and Conference Proceedings.
    """
    @autoassign
    def __init__(self, groups=None, pen_val=1): pass


class MultiTaskLasso(WithFlavorPenSeqConfig):
    """
    The multi-task lasso (including adaptive and non-convex flavors) for multiple response coefficients.

    pen_val * sum_{j=1}^{n_features} w_r ||coef(j, :)||_2
    sum_{j=1}^{n_features} non_convex_{pen_val}(||coef(j, :)||_2)

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    weights: None, array-like shape (n_features, )
        (Optional) Weights for each feature.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.
    """
    @autoassign
    def __init__(self, pen_val=1, weights=None, flavor=None): pass

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 multi_task=True,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


class NuclearNorm(WithFlavorPenSeqConfig):
    """
    Nuclear norm, adaptive nuclear norm or non-convex nuclear norm.

    pen_val * ||coef||_*
    pen_val * sum_{j} w_j sigma_j(coef)
    sum_{j} non-convex_{pen_val}(sigma_j(coef))

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    weights: None, array-like
        (Optional) Weights for each term in the penalty.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.
    """

    @autoassign
    def __init__(self, pen_val=1, weights=None, flavor=None): pass

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 nuc=True,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


class FusedLasso(WithFlavorPenSeqConfig):
    """
    The graph fused lasso also known as graph trend filtering. The fused lasso (i.e. total-variation 1 penalty) is a special case when the graph is a chain graph. This penalty includes higher order trend filtering and can represent the adaptive an non-convex versions of the graph fused lasso.

    Note this is NOT the sparse fused lasso.

    The standard fused lasso (chain graph) is given by:
    pen_val * sum_{j=1}^{n_features -1} |coef_{j+1} - coef_j|

    The graph fused lasso is given by:
    pen_val * sum_{(ij) in edgelist} w_{(ij)} |coef_i - coef_j|
    sum_{(ij) in edgelist} non-convex_{pen_val}(|coef_i - coef_j|)

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    edgelist: str, array-like (n_edges, 2)
        The graph's edgelist. If edgelist='chain' then this is the TV-1 penalty.

    order: int
        The order of the trend filtering difference.

    weights: None, array-like
        (Optional) Weights for edge. If edgelish='chain', this should be length n_features - 1.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.

    References
    ----------
    Wang, Y.X., Sharpnack, J., Smola, A. and Tibshirani, R., 2015, February. Trend filtering on graphs. In Artificial Intelligence and Statistics (pp. 1042-1050). PMLR.

    Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.


    Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. and Knight, K., 2005. Sparsity and smoothness via the fused lasso. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(1), pp.91-108.
    """
    @autoassign
    def __init__(self, pen_val=1, edgelist='chain', order=1,
                 weights=None, flavor=None): pass


class GeneralizedLasso(WithFlavorPenSeqConfig):
    """
    The generalized lasso including the adaptive a non-convex versions.

    pen_val * ||mat @ coef||_1
    sum_{r=1}^{p} non-convex_{pen_val}(|mat(r, :).T @ coef|)

    Parameters
    ----------
    pen_val: float
        The penalty parameter value.

    mat: array-like, shape (p, n_features)
        The transformation matrix.

    weights: None, array-like
        (Optional) Weights for each term in mat @ coef.

    flavor: None, PenaltyFlavor
        (Optional) Which flavor of the penalty to use e.g. adaptive, non-convex direct, or non-convex LLA.

    References
    ----------
    Tibshirani, R.J. and Taylor, J., 2011. The solution path of the generalized lasso. The annals of statistics, 39(3), pp.1335-1371.

    Ali, A. and Tibshirani, R.J., 2019. The generalized lasso problem and uniqueness. Electronic Journal of Statistics, 13(2), pp.2307-2347.
    """
    @autoassign
    def __init__(self, pen_val=1, mat=None, weights=None, flavor=None): pass

########################
# ElasticNet penalties #
########################

# TODO: maybe add ridge weights?
class ElasticNet(ElasticNetConfig):
    """
    Represents the ElasticNet penalty

    pen_val * mix_val ||coef||_1 + 0.5 * pen_val * (1 - mix_val) * ||coef||_2^2

    The Lasso may have weights (though not the ridge at this time) or may be flavored.
    We define the non-convex elastic net as

    non-convex_{pen_val * mix_val} (coef) + 0.5 * pen_val * (1 - mix_val) * ||coef||_2^2

    Parameters
    ----------
    pen_val: float
        The penalty strength.

    mix_val: float
        The mixing value between 0 and 1.

    lasso_weights: None, array-like
        (Optional) Weights for the Lasso penalty.

    lasso_flavor: None, FlavorConfig
        (Optional) Flavor for the lasso penalty.

    ridge_weights: None, array-like
        (Optional) Weights for the ridge penalty.
    """
    @autoassign
    def __init__(self, pen_val=1, mix_val=0.5,
                 lasso_weights=None, lasso_flavor=None,
                 ridge_weights=None): pass

    def _get_sum_configs(self):
        lasso_config = Lasso(pen_val=self.pen_val * self.mix_val,
                             weights=self.lasso_weights,
                             flavor=self.lasso_flavor)

        ridge_config = Ridge(pen_val=self.pen_val * (1 - self.mix_val),
                             weights=self.ridge_weights)

        return lasso_config, ridge_config

    def get_sum_names(self):
        return ['lasso', 'ridge']


# TODO: add default weights
# TODO: add default to infimal for overlapping
# TODO: should we allow ridge weights?
class GroupElasticNet(ElasticNetConfig):
    """
    Represents the group ElasticNet penalty

    pen_val * mix_val * gruop_lasso(coef; groups) + pen_val * (1 - mix_val) * ridge(coef)

    non_convex-group_{pen_val * mix_val}(coef; groups) + pen_val * (1 - mix_val) * ridge(coef)

    Parameters
    ----------
    groups: list, None
        Indices of the groups. If None, then all features are put in a single group.

    pen_val: float
        The penalty strength.

    mix_val: float
        The mixing value between 0 and 1.

    lasso_weights: None, array-like
        (Optional) Weights for the Lasso.

    lasso_flavor: None, FlavorConfig
        (Optional) Flavor for the lasso penalty.

    ridge_weights: None, array-like
        (Optional) Weights for the ridge penalty.
    """
    @autoassign
    def __init__(self, groups=None,
                 pen_val=1, mix_val=0.5,
                 lasso_weights=None, lasso_flavor=None,
                 ridge_weights=None): pass

    def _get_sum_configs(self):
        lasso_config = GroupLasso(groups=self.groups,
                                  pen_val=self.pen_val * self.mix_val,
                                  weights=self.lasso_weights,
                                  flavor=self.lasso_flavor)

        ridge_config = Ridge(pen_val=self.pen_val * (1 - self.mix_val),
                             weights=self.ridge_weights)

        return lasso_config, ridge_config

    def get_sum_names(self):
        return ['lasso', 'ridge']


# TODO: should we allow ridge weights?
class MultiTaskElasticNet(ElasticNetConfig):
    """
    Represents the group MultiTask ElasticNet penalty

    pen_val * mix_val * multi-task(coef) + pen_val * (1 - mix_val) * ridge(coef)

    non-convex-multi-task_{pen_val * mix_val}(coef) + pen_val * (1 - mix_val) * ridge(coef)

    Parameters
    ----------
    pen_val: float
        The penalty strength.

    mix_val: float
        The mixing value between 0 and 1.

    lasso_weights: None, array-like
        (Optional) Weights for the Lasso.

    lasso_flavor: None, FlavorConfig
        (Optional) Flavor for the lasso penalty.

    ridge_weights: None, array-like
        (Optional) Weights for the ridge penalty.
    """
    @autoassign
    def __init__(self, pen_val=1, mix_val=0.5,
                 lasso_weights=None, lasso_flavor=None,
                 ridge_weights=None): pass

    def _get_sum_configs(self):
        lasso_config = MultiTaskLasso(pen_val=self.pen_val * self.mix_val,
                                      weights=self.lasso_weights,
                                      flavor=self.lasso_flavor)

        ridge_config = Ridge(pen_val=self.pen_val * (1 - self.mix_val),
                             weights=self.ridge_weights)

        return lasso_config, ridge_config

    def get_sum_names(self):
        return ['lasso', 'ridge']

# TODO: add default weights for group
class SparseGroupLasso(ElasticNetConfig):
    """
    Represents the sparse group lasso penalty.

    pen_val * mix_val * ||coef||_1 + pen_val * (1 - mix_val) * group_lasso(coef; groups)

    non-convex_{pen_val*mix_val}(coef) + pen_val * (1 - mix_val) * group_lasso(coef; groups)

    Parameters
    ----------
    groups: list, None
        Indices of the groups. If None, then all features are put in a single group.

    pen_val: float
        The penalty strength.

    mix_val: float
        The mixing value between 0 and 1.

    sparse_weights: None, array-like shape (n_features, ) or (n_features, n_responses)
        (Optional) Weights for the entrywise lasso.

    sparse_flavor: None, FlavorConfig
        (Optional) Flavoring for the entrywise penalty.

    group_weights: None, array-like shape (n_groups)
        (Optional) Weights for the group penalty.

    sparse_flavor: None, FlavorConfig
        (Optional) Flavoring for the group penalty.

    References
    ----------
    Simon, N., Friedman, J., Hastie, T. and Tibshirani, R., 2013. A sparse-group lasso. Journal of computational and graphical statistics, 22(2), pp.231-245.
    """

    @autoassign
    def __init__(self, groups=None, pen_val=1, mix_val=0.5,
                 sparse_weights=None, sparse_flavor=None,
                 group_weights=None, group_flavor=None): pass

    def _get_sum_configs(self):
        sparse_config = Lasso(pen_val=self.pen_val * self.mix_val,
                              weights=self.sparse_weights,
                              flavor=self.sparse_flavor)

        group_config = GroupLasso(pen_val=self.pen_val * (1 - self.mix_val),
                                  groups=self.groups,
                                  weights=self.group_weights,
                                  flavor=self.group_flavor)

        return sparse_config, group_config

    def get_sum_names(self):
        return ['sparse', 'group']

###################################
# Other overlapping sum penalties #
###################################

# TODO: add this
# class SparseFusedLasso:
#     def __init__(self, fused=FusedLasso(), sparse=Lasso()): pass

#########################
# Infimal Sum penalties #
#########################
# TODO: add these
# TODO: think about best order for penalties

# class LowRankPlusSparse(InfimalSumConfig):
#     @autoassign
#     def __init__(self, rank=NuclearNorm(), sparse=Lasso()): pass


# class LowRankPlusRowSparse(InfimalSumConfig):
#     @autoassign
#     def __init__(self, rank=NuclearNorm(), sparse=MultiTaskLasso()): pass


# class RowPlusEntrywiseSparse(InfimalSumConfig):
#     @autoassign
#     def __init__(self, row=MultiTaskLasso(), sparse=Lasso()): pass


##############################
# On the fly penalty configs #
##############################
# these are exactly the configs, but we rename them/store them here


class SeparableSum(SeparableSumConfig):
    pass


class InfimalSum(InfimalSumConfig):
    pass


class OverlappingSum(OverlappingSumConfig):
    pass


#########
# utils #
#########

def get_penalty_config(config):
    """
    Gets the penalty config object.

    Parameters
    ----------
    loss: str, LossConfig, or TunerConfig
        The penalty.

    Output
    ------
    config: LossConfig:
        The penalty config object.
    """
    if type(config) != str:
        return config
    else:

        return penalty_str2obj[config.lower()]


penalty_str2obj = {'none': NoPenalty(),
                   'ridge': Ridge(),
                   'gen_ridge': GeneralizedRidge(),
                   'lasso': Lasso(),

                   'group': GroupLasso(),
                   'exclusive_group': ExclusiveGroupLasso(),

                   'multi_task': MultiTaskLasso(),
                   'nuc': NuclearNorm(),
                   'fused': FusedLasso(),
                   'gen_lasso': GeneralizedLasso(),

                   'enet': ElasticNet(),
                   'group_enet': GroupElasticNet(),
                   'multi_task_enet': MultiTaskElasticNet(),

                   'sparse_group': SparseGroupLasso(),

                   # 'sparse_fused': SparseFusedLasso(),
                   # 'low_rank_plus_sparse': LowRankPlusSparse(),
                   # 'low_rank_plus_row_sparse': LowRankPlusRowSparse(),
                   # 'row_plus_entrywise_sparse': RowPlusEntrywiseSparse()
                   }

avail_penalties = list(penalty_str2obj.keys())
