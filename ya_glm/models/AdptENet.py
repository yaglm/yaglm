from ya_glm.base.GlmAdptPen import GlmAdptPen, GlmAdptPenCV
from ya_glm.base.GlmENetCV import GlmENetCVMixin

from ya_glm.PenaltyConfig import AdptPenalty
from ya_glm.loss.LossMixin import LossMixin

from ya_glm.models.ENet import ENet, ENetCV

from ya_glm.utils import lasso_and_ridge_from_enet
from ya_glm.processing import check_estimator_type
from ya_glm.autoassign import autoassign


class AdptENet(LossMixin, GlmAdptPen):
    """
    A GLM with an Adpative ElasticNet-like penalty. Note the Lasso can be: Lasso, group Lasso, multi-task Lasso, or nuclear norm.

    Here the Lasso weights are adpative and are set as in AdptLasso

    Parameters
    ----------
    loss: str, ya_glm.LossConfig.LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See ya_glm.LossConfig for available loss functions.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    pen_val: float
        The multiplicative penalty strength (corresponds to lambda in glmnet).

    l1_ratio: float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
            ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` itis an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
            combination of L1 and L2.

    lasso_weights: None, array-like
        Optional weights to put on each term in the penalty.

    groups: None, list of ints
        Optional groups of variables. If groups is provided then each element in the list should be a list of feature indices. Variables not in a group are not penalized.

    multi_task: bool
        Use a multi-task Lasso for the coefficient matrix of multiple response GLM. This is the L1 to L2 norm (sum of euclidean norms of the rows).

    nuc: bool
        Use a nuclear norm penalty (sum of the singular values) for the coefficient matrix of multiple response GLM.

    ridge_weights: None, array-like shape (n_featuers, )
        (Optional) Features weights for the ridge peanlty.

    tikhonov: None, array-like (K, n_features)
        (Optional) Tikhonov matrix for the ridge penalty. Both tikhonov and ridge weights cannot be provided at the same time.

    init: str, dict, estimator.
        If init='default', will use ENetCV as the initializer.
        If init is a dict, will return self.init. If init is an estimator that is already fit, it will NOT be refit on the new data. If init is a dict with 'adpt_weights' the the estimator will use exactly these adpative weights.

    adpt_func: str
        The concave function whose gradient is used to obtain the adpative weights from the initial coefficient. See ya_glm.opt.penalty.concave_penalty.

    adpt_func_kws: dict
        Keyword arguments to the adpative function e.g. q for the Lq norm.

    pertub_init: str, float
        How to perturb the initial coefficient i.e. we evaluate the adpative function's gradient at abs(init_coef) + pertub_init. If pertub_init='n_samples', then we use 1/n_samples. This perturbation is useful, for example, when the init coefficient has exact zeros and adpt_func='log'.


    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Standardization means mean centering and scaling each column by its standard deviation. For the group lasso penalty an additional scaling is applied that scales each variable by 1 / sqrt(group size). Putting each variable on the same scale makes sense for fitting penalized models. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    solver: str, ya_glm.GlmSolver
        The solver used to solve the penalized GLM optimization problem. If this is set to 'default' we try to guess the best solver. Otherwise a custom solver can be provided by specifying a GlmSolver object.

    Attributes
    ----------
    coef_: array-like, shape (n_features, ) or (n_features, n_responses)
        The fitted coefficient vector or matrix (for multiple responses).

    intercept_: None, float or array-like, shape (n_features, )
        The fitted intercept.

    classes_: array-like, shape (n_classes, )
        A list of class labels known to the classifier.

    opt_data_: dict
        Data output by the optimization algorithm.
    """
    @autoassign
    def __init__(self, loss='lin_reg', fit_intercept=True,

                 pen_val=1, l1_ratio=0.5,
                 groups=None, multi_task=False, nuc=False,
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None,

                 adpt_func='log',  # TODO: is this the name we want?
                 adpt_func_kws={},
                 init='default',
                 pertub_init='n_samples',

                 standardize=False, solver='default'):
        pass

    def _get_penalty_config(self):
        """
        Gets the penalty config. If self.init has an 'adpt_weights' key, then the lasso_weights are set using these.

        Output
        ------
        penalty: ya_glm.PenaltyConfig.AdptPenalty
            A penalty config object.
        """

        lasso_pen_val, ridge_pen_val = \
            lasso_and_ridge_from_enet(pen_val=self.pen_val,
                                      l1_ratio=self.l1_ratio)

        config = AdptPenalty(adpt_func=self.adpt_func,
                             adpt_func_kws=self.adpt_func_kws,
                             pertub_init=self.pertub_init,

                             lasso_pen_val=lasso_pen_val,
                             groups=self.groups,
                             multi_task=self.multi_task,
                             nuc=self.nuc,
                             ridge_pen_val=ridge_pen_val,
                             ridge_weights=self.ridge_weights,
                             tikhonov=self.tikhonov
                             )

        if self.has_preset_adpt_weights():
            config.set_adpt_weights(self.init['adpt_weights'])

        return config

    @property
    def _primary_penalty_type(self):
        if self.l1_ratio > 0:
            return 'lasso'
        else:
            return 'ridge'

    def _get_default_init(self):
        """
        Output
        -------
        est: LassoCV()
            The default initializer object.
        """

        est = ENet(loss=self.loss,
                   fit_intercept=self.fit_intercept,

                   groups=self.groups,
                   multi_task=self.multi_task,
                   nuc=self.nuc,
                   ridge_pen_val=self.ridge_pen_val,
                   ridge_weights=self.ridge_weights,
                   tikhonov=self.tikhonov,

                   standardize=self.standardize,
                   solver=self.solver
                   )

        return ENetCV(estimator=est)


class AdptENetCV(GlmENetCVMixin, GlmAdptPenCV):
    """
    Tunes an Adpative ElasticNet-like penalized GLM using cross-validation. Makes use of a path algorithm if the solver object has one available. One or both of pen_val and l1_ratio are tuned.

    Note the initializer is fit before running cross-validation so the same adpat_weights are used for each CV fold.

    Parameters
    ----------
    estimator: ya_glm.models.AdptENet
        The base AdptENet estimator to be tuned with cross-validation. The pen_val and/or l1_ratio values are tuned.

    cv: int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

    cv_select_metric: None, str
        Which metric to use for select the best tuning parameter if multiple metrics are computed.

    cv_scorer: None, callable(est, X, y) -> dict or float
        A function for evaluating the cross-validation fit estimators. If this returns a dict of multiple scores then cv_select_metric determines which metric is used to select the tuning parameter.

    cv_n_jobs: None, int
        Number of jobs to run in parallel.

    cv_verbose: int
        Amount of printout during cross-validation.

    cv_pre_dispatch: int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel execution

    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence.

    pen_vals: None, array-like
        (Optional) User provided penalty value sequence. The penalty sequence should be monotonicly decreasing so the homotopy path algorithm works propertly.

    pen_min_mult: float
        Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    l1_ratio: float, str, list
        The l1_ratio value to use. If a float is provided then this parameter is fixed and not tuned over. If l1_ratio='tune' then the l1_ratio is tuned over using an automatically generated tuning parameter sequence. Alternatively, the user may provide a list of l1_ratio values to tune over.

    n_l1_ratio_vals: int
        Number of l1_ratio values to tune over. The l1_ratio tuning sequence is a logarithmically spaced grid of values between 0 and 1 that has more values close to 1.

    l1_ratio_min:
        The smallest l1_ratio value to tune over.

    Attributes
    ----------
    best_estimator_:
        The fit estimator with the parameters selected via cross-validation.

    cv_results_: dict
        The cross-validation results.

    best_tune_idx_: int
        Index of the best tuning parameter. This index corresponds to the list returned by get_tuning_sequence().

    best_tune_params_: dict
        The best tuning parameters.

    cv_data_: dict
        Additional data about the CV fit e.g. the runtime.
    """
    @autoassign
    def __init__(self,
                 estimator=AdptENet(),

                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs',

                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log',
                 l1_ratio=0.5,
                 n_l1_ratio_vals=10,
                 l1_ratio_min=0.1
                 ): pass

    def _check_base_estimator(self):
        check_estimator_type(self.estimator, AdptENet)
