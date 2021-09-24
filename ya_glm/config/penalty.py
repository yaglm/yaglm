from ya_glm.config.base_penalty import PenaltyConfig, WithPenSeqConfig, \
    WithFlavorPenSeqConfig

from ya_glm.pen_max.ridge import get_ridge_pen_max
from ya_glm.pen_max.lasso import get_lasso_pen_max

from ya_glm.autoassign import autoassign


class NoPenalty(PenaltyConfig):
    """
    Represents no penalty.
    """
    def get_func_info(self):
        return {'smooth': True, 'proximable': True, 'lin_proximable': True}


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
    def __init__(self, pen_val=1, weights=None): pass

    def get_func_info(self):
        return {'smooth': True, 'proximable': True, 'lin_proximable': True}

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None):

        return get_ridge_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight,
                                 targ_ubd=1,
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

    def get_func_info(self):
        return {'smooth': True, 'proximable': False, 'lin_proximable': True}


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

    def get_func_info(self):
        return {'smooth': False, 'proximable': True, 'lin_proximable': True}

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


# TODO: add default weights
class GroupLasso(WithFlavorPenSeqConfig):
    """
    Group penalty e.g. group lasso, adaptive group lasso, or group non-convex.

    pen_val * sum_{g in groups} weights_g ||coef_g||_2
    sum_{g in groups} non-convex_{pen_val}(||coef_g||_2)

    Parameters
    ----------
    groups: list
        Indices of the groups.

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
    def __init__(self, groups, pen_val=1, weights=None, flavor=None): pass

    def get_func_info(self):
        return {'smooth': False, 'proximable': True, 'lin_proximable': True}

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
class ExclusiveGroupLasso(WithFlavorPenSeqConfig):
    """
    Exclusive group lasso.

    pen_val * sum_{g in groups} ||coef_g||_1^2

    Parameters
    ----------
    groups: list
        Indices of the groups.

    pen_val: float
        The penalty parameter value.

    References
    ----------
    Campbell, F. and Allen, G.I., 2017. Within group variable selection through the exclusive lasso. Electronic Journal of Statistics, 11(2), pp.4220-4257.

    Zhou, Y., Jin, R. and Hoi, S.C.H., 2010, March. Exclusive lasso for multi-task feature selection. In Proceedings of the thirteenth international conference on artificial intelligence and statistics (pp. 988-995). JMLR Workshop and Conference Proceedings.
    """
    @autoassign
    def __init__(self, groups, pen_val=1): pass

    def get_func_info(self):
        return {'smooth': False, 'proximable': True, 'lin_proximable': True}


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

    def get_func_info(self):
        return {'smooth': False, 'proximable': True, 'lin_proximable': True}

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

    def get_func_info(self):
        return {'smooth': False, 'proximable': True, 'lin_proximable': True}

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

    def get_func_info(self):
        # TODO: I'm pretty sure TV-1 is proximable
        return {'smooth': False, 'proximable': False, 'lin_proximable': True}


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

    def get_func_info(self):
        # TODO: I'm pretty sure TV-1 is proximable
        return {'smooth': False, 'proximable': False, 'lin_proximable': True}

# TODO: add multi-penalties
# class ElasticNet: pass
# class GroupElasticNet: pass
# class MultiTaskElasticNet: pass
# class SparseGroupLasso: pass
# class SparseFusedLasso: pass


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

                   # TODO: how to handle the positional init argument?
                   'group': GroupLasso(groups=None),
                   'exclusive_group': ExclusiveGroupLasso(groups=None),

                   'multi_task': MultiTaskLasso(),
                   'nuc': NuclearNorm(),
                   'fused': FusedLasso(),
                   'gen_lasso': GeneralizedLasso()
                   }

avail_penalties = list(penalty_str2obj.keys())
