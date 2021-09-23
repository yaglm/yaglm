from functools import partial

from ya_glm.config.base import safe_get_config
from ya_glm.config.penalty_base import PenaltyConfig, PenSeqTunerMixin, \
    FlavoredMixin

from ya_glm.pen_max.ridge import get_ridge_pen_max
from ya_glm.pen_max.lasso import get_lasso_pen_max

from ya_glm.transforms import entrywise_abs_transform,\
    multi_task_lasso_transform, group_transform, sval_transform,\
    fused_lasso_transform, generalized_lasso_transform
from ya_glm.autoassign import autoassign


class NoPenalty(PenaltyConfig):
    """
    Represents no penalty.
    """
    def set_tuning_values(self, *args, **kws):
        """
        This of course does nothing
        """
        pass

    def is_smooth(self):
        return True


class Ridge(PenSeqTunerMixin, PenaltyConfig):
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

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None):

        return get_ridge_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight,
                                 targ_ubd=1,
                                 norm_by_dim=True)

    def is_smooth(self):
        return True


class GeneralizedRidge(PenSeqTunerMixin, PenaltyConfig):
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

    @property
    def is_proximable(self):
        return False

    def is_smooth(self):
        return True


class Lasso(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def get_non_smooth_transforms(self):
        return entrywise_abs_transform


# TODO: add default weights
class GroupLasso(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 groups=self.groups,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)

    def get_non_smooth_transforms(self):
        return partial(group_transform, groups=self.groups)


# TODO: add flavor
# TODO: add weights -- should we have both entrywise and group?
# perhaps default gives group lasso weights
class ExclusiveGroupLasso(PenSeqTunerMixin, PenaltyConfig):
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


class MultiTaskLasso(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def get_non_smooth_transforms(self):
        return multi_task_lasso_transform

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):

        return get_lasso_pen_max(X=X, y=y, loss=loss,
                                 weights=self.weights,
                                 multi_task=True,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)


class NuclearNorm(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def get_non_smooth_transforms(self):
        return sval_transform


class FusedLasso(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def get_non_smooth_transforms(self):
        return partial(fused_lasso_transform,
                       edgelist=self.edgelist,
                       order=self.order)

    @property
    def is_proximable(self):
        # TODO: I'm pretty sure TV-1 is proximable
        return False


class GeneralizedLasso(FlavoredMixin, PenSeqTunerMixin, PenaltyConfig):
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

    def get_non_smooth_transforms(self):
        return partial(generalized_lasso_transform, mat=self.mat)

    @property
    def is_proximable(self):
        return False

# TODO: add multi-penalties
# class ElasticNet: pass
# class GroupElasticNet: pass
# class MultiTaskElasticNet: pass
# class SparseGroupLasso: pass
# class SparseFusedLasso: pass


def get_penalty_config(config):
    """
    Gets the penalty config object. If a tuned penalty is provided this will return the base penalty config.

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
        return safe_get_config(config)
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
