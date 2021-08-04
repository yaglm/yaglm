from ya_glm.base.GlmCvxPen import GlmCvxPen
from ya_glm.base.GlmCV import GlmCV, SinglePenSeqSetterMixin

from ya_glm.loss.LossMixin import LossMixin
from ya_glm.PenaltyConfig import ConvexPenalty, PenaltySequence

from ya_glm.cv.RunCVMixin import RunCVGridOrPathMixin

from ya_glm.processing import check_estimator_type
from ya_glm.autoassign import autoassign


class Lasso(LossMixin, GlmCvxPen):
    """
    A GLM with a lasso-like penalty e.g. Lasso, group Lasso, multi-task Lasso, or nuclear norm.

    Parameters
    ----------
    loss: str, ya_glm.LossConfig.LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See ya_glm.LossConfig for available loss functions.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    pen_val: float
        The Lasso penalty value.

    lasso_weights: None, array-like
        Optional weights to put on each term in the penalty.

    groups: None, list of ints
        Optional groups of variables. If groups is provided then each element in the list should be a list of feature indices. Variables not in a group are not penalized.

    multi_task: bool
        Use a multi-task Lasso for the coefficient matrix of multiple response GLM. This is the L1 to L2 norm (sum of euclidean norms of the rows).

    nuc: bool
        Use a nuclear norm penalty (sum of the singular values) for the coefficient matrix of multiple response GLM.

    ridge_pen_val: None, float
        (Optional) Penalty strength for an optional ridge penalty.

    ridge_weights: None, array-like shape (n_featuers, )
        (Optional) Features weights for the ridge peanlty.

    tikhonov: None, array-like (K, n_features)
        (Optional) Tikhonov matrix for the ridge penalty. Both tikhonov and ridge weights cannot be provided at the same time.

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

                 pen_val=1, lasso_weights=None,
                 groups=None, multi_task=False, nuc=False,
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None,

                 standardize=False, solver='default'):
        pass

    def _get_penalty_config(self):
        """
        Gets the penalty config.

        Output
        ------
        penalty: ya_glm.PenaltyConfig.ConvexPenalty
            A penalty config object.
        """

        return ConvexPenalty(lasso_pen_val=self.pen_val,
                             lasso_weights=self.lasso_weights,
                             groups=self.groups,
                             multi_task=self.multi_task,
                             nuc=self.nuc,
                             ridge_pen_val=self.ridge_pen_val,
                             ridge_weights=self.ridge_weights,
                             tikhonov=self.tikhonov
                             )

    @property
    def _primary_penalty_type(self):
        return 'lasso'


class LassoCV(SinglePenSeqSetterMixin, RunCVGridOrPathMixin, GlmCV):
    """
    Tunes a Lasso-like penalized GLM using cross-validation. Makes use of a path algorithm if the solver object has one available.

    Parameters
    ----------
    estimator: ya_glm.models.Lasso
        The base Lasso estimator to be tuned with cross-validation. Only the pen_val parameter is tuned.

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
                 estimator=Lasso(),

                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs',

                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log'
                 ):
        pass

    def _get_penalty_seq_config(self, estimator):
        """
        Gets the penalty sequence config for tuning the penalty parameter.

        Output
        ------
        penalty: ya_glm.PenaltyConfig.PenaltySequence
            A penalty sequence config object.
        """

        if not hasattr(self, 'pen_val_seq_'):
            raise RuntimeError("pen_val_seq_ has not yet been set")

        penalty = estimator._get_penalty_config()
        return PenaltySequence(penalty=penalty,
                               lasso_pen_seq=self.pen_val_seq_)

    def _check_base_estimator(self):
        check_estimator_type(self.estimator, Lasso)
