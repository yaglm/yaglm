import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, _check_sample_weight, \
    FLOAT_DTYPES

from ya_glm.autoassign import autoassign
from ya_glm.processing import process_X, deprocess_fit, process_init_data
from ya_glm.utils import fit_if_unfitted, get_coef_and_intercept, \
    is_str_and_matches, get_shapes_from
from ya_glm.config.utils import is_flavored

from ya_glm.config.loss import get_loss_config
from ya_glm.config.constraint import get_constraint_config
from ya_glm.config.flavor import get_flavor_config
from ya_glm.solver.default import get_solver, maybe_get_lla
from ya_glm.config.base import TunerConfig, safe_get_config
from ya_glm.tune.backend import run_fit_and_score_jobs
from ya_glm.tune.combined_tuner import PenaltyPerLossFlavorTuner


class BaseGlm(BaseEstimator):
    """
    Base class for GLMs.

    Parameters
    ----------
    loss: str, LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See ya_glm.LossConfig for available loss functions.

    penalty: None, PenaltyConfig
        The penalty config object specifying the penalty e.g. Lasso, Ridge, ...

    constraint: None, ConstraintConfig
        (Optional) The constraint config object e.g. Positive, Isotonic, ...

    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Standardization means mean centering and scaling each column by its standard deviation. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    solver: str, ya_glm.GlmSolver
        The solver used to solve the penalized GLM optimization problem. If this is set to 'default' we try to guess the best solver. Otherwise a custom solver can be provided by specifying a GlmSolver object.

    initializer: str, dict, Estimator
        Specifies the initial estimator to use for penalties that require an initial estimator e.g. adaptive and non-convex LLA penalties. If str, must be one of ['default', 'zero']. If init='default', will infer a reasonable default penalty to use. If 'init'='zero', will initialize from zero. If an estimator is provided, will fit the estimator to the training data (unless it is already fit) then use the estimtor's fit coefficient. If a dict is provided, it must include the key 'coef' for the initializer coefficient.

    inferencer: None, Inferencer
        (Optional) An object that runs statistical inference procedures on the fitted estimator.

    Attributes
    ----------
    coef_: array-like, shape (n_features, ) or (n_features, n_responses)
        The fitted coefficient vector or matrix (for multiple responses).

    intercept_: None, float or array-like, shape (n_features, )
        The fitted intercept.

    classes_: array-like, shape (n_classes, )
        A list of class labels known to the classifier.

    opt_info_: dict
        Data output by the optimization algorithm.

    fit_penalty_: PenaltyConfig
        The penalty config that was fit e.g. stores the adapative weights.

    fit_loss_:
        The loss config that was fit.

    fit_constraint_:
        The constraint config that was fit.
    """
    @autoassign
    def __init__(self, loss='lin_reg',
                 penalty=None, constraint=None,
                 standardize=True, fit_intercept=True,
                 solver='default',
                 initializer='default',
                 inferencer=None
                 ):
        pass

    @property
    def _estimator_type(self):
        """
        Type of the estimator.

        Output
        ------
        _estimator_type: str
            Either 'regressor' or 'classifier'.
        """
        loss_config = get_loss_config(self.loss)
        return loss_config._estimator_type

    @property
    def _is_tuner(self):
        """
        Whether or not this estimator object selects the tuning parameters.

        Output
        ------
        is_tuner: bool
            Yes this is a tuning object
        """
        raise NotImplementedError

    def get_initializer(self, X, y, pre_pro_out, sample_weight=None):
        """
        Possibly performs prefitting routines e.g. fitting an initial estimator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The raw covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The raw response data.

        pre_pro_out: dict
            Preprocessing output.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        init_data_pro, init_est

        init_data_pro: None, dict
            The processed initialization data.

        init_est: None, Estimator
            The initializer estimator if one was fit.
        """

        ######################################
        # Initializer for flavored penalties #
        ######################################
        init_data_pro = None
        init_est = None

        if is_flavored(self.penalty):

            ##############################################
            # Get initializer coefficient/intercept data #
            ##############################################
            init_est = None

            if is_str_and_matches(self.initializer, 'default'):
                # get the default initial estimator for this flavor
                init = get_flavor_config(self.penalty.flavor)\
                    .get_default_init(est=self)
            else:
                init = self.initializer

            # user provided initial values
            if isinstance(init, dict):
                init_data = deepcopy(self.initializer)

            # initialize at zero
            elif is_str_and_matches(init, 'zero'):
                coef_shape, intercept_shape = get_shapes_from(X=X, y=y)

                coef = np.zeros(coef_shape)

                if intercept_shape[0] == 0:
                    intercept = 0
                else:
                    intercept = np.zeros(intercept_shape)

                init_data = {'coef': coef, 'intercept': intercept}

            # initialize from an estimator fit to the data
            else:

                # possibly fit this estimator if it has not already been fit
                init_est = fit_if_unfitted(init, X=X, y=y,
                                           sample_weight=sample_weight)

                coef, intercept = get_coef_and_intercept(init_est, copy=True,
                                                         error=True)

                init_data = {'coef': coef, 'intercept': intercept}

            # process initializer data and return it
            init_data_pro = \
                process_init_data(init_data=init_data,
                                  pre_pro_out=pre_pro_out)

            # adaptive Lasso may need to know the number of training samples
            init_data_pro['n_samples'] = X.shape[0]

        return init_data_pro, init_est

    def _validate_data(self, X, y, sample_weight=None, accept_sparse=True):
        """
        Validates the X/y data. This should not change the raw input data, but may reformat the data (e.g. convert pandas to numpy).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.
        """
        X = check_array(X, accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # make sure y is numpy and of same dtype as X
        # TODO: do we actually want this for log_reg/multinomial?
        y = check_array(y, ensure_2d=False)

        # make sure 1d input is actually a vector
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        # make sure X, y have same number of samples
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows!")

        return X, y, sample_weight

    def preprocess(self, X, y, sample_weight=None, copy=True, check_input=True):
        """
        Preprocesses the data for fitting.

        If standardize=True then the coulmns are scaled to be unit norm. If additionally fit_intercept=True, the columns of X are first mean centered before scaling.

        For the group lasso penalty an additional scaling is applied that scales each variable by 1 / sqrt(group size).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        copy: bool
            Whether or not to copy the X/y arrays or modify them in place.

        Output
        ------
        X_pro, y_pro, pre_pro_out

        X_pro: array-like, shape (n_samples, n_features)
            The possibly transformed covariate data.

        y_pro: array-like, shape (n_samples, )
            The possibly transformed response data.

        sample_weight_pro: None or array-like,  shape (n_samples,)
            The processed sample weights. Ensures sum(sample_weight) = n_samples. Possibly incorporate class weights.

        pro_pro_out: dict
            Data from preprocessing e.g. X_center, X_scale.
        """

        if sample_weight is not None:
            if copy:
                sample_weight = sample_weight.copy()

        # possibly standarize X
        X, out = process_X(X,
                           standardize=self.standardize,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight,
                           copy=copy,
                           check_input=check_input,
                           accept_sparse=True)

        # subclass should implement this
        # possibly process y
        y, sample_weight, y_out = \
            self._process_y(X=X, y=y,
                            sample_weight=sample_weight,
                            copy=copy)

        # ensure sum(sample_weight) = n_samples
        if sample_weight is not None:
            sample_weight *= len(sample_weight) / sample_weight.sum()

        out.update(y_out)

        return X, y, sample_weight, out

    def _fit_from_configs(self, configs, solver, pre_pro_out, solver_init,
                          X, y, X_pro, y_pro,
                          sample_weight=None, sample_weight_pro=None):
        """
        TODO: document
        """

        ##########
        # solve! #
        ##########

        # setup solver
        solver.setup(X=X_pro,
                     y=y_pro,
                     fit_intercept=self.fit_intercept,
                     sample_weight=sample_weight_pro,
                     **configs,  # loss, penalty, constraint
                     )

        fit_out, _,  opt_info = solver.solve(**solver_init)

        ##################
        # post procesing #
        ##################

        # set the fit coefficient e.g. undo preprocessing scaling
        self._set_fit(fit_out=fit_out,
                      pre_pro_out=pre_pro_out,
                      configs=configs,
                      opt_info=opt_info)

        # run any post fitting statistical inference e.g. estimate number of DoF
        self.run_postfit_inference(X=X, y=y,
                                   sample_weight=sample_weight)

        return self

    def _set_fit(self, fit_out, pre_pro_out,
                 configs=None,
                 opt_info=None):
        """
        Sets the fit from the ouptut of the optimization algorithm.
        For example, this undoes any centering and scaling we have performed on the data so the fitted coefficient matches the raw input data.

        Parameters
        ----------
        fit_out: dict
            Contains the output of solve e.g.
            fit_out['coef'], fit_out['intercept'], fit_out['opt_info']

        pre_pro_out: None, dict
            Output of preprocess.

        configs: None, dict
            (Optional) A dict containing the loss, penalty and constraint configs corresponding to this fit. Should have keys ['loss', 'penalty', 'constraint'].

        opt_info: None, dict.
            (Optional) Optimization information output.

        """
        coef = fit_out['coef']
        intercept = fit_out.get('intercept', None)

        self.coef_, self.intercept_ = \
            deprocess_fit(coef=coef,
                          intercept=intercept,
                          pre_pro_out=pre_pro_out,
                          fit_intercept=self.fit_intercept)

        if not self.fit_intercept:
            self.intercept_ = None

        # for classification models
        if 'label_encoder' in pre_pro_out:
            self.label_encoder_ = pre_pro_out['label_encoder']
            self.classes_ = self.label_encoder_.classes_

        elif 'label_binarizer' in pre_pro_out:
            self.label_binarizer_ = pre_pro_out['label_binarizer']
            self.classes_ = self.label_binarizer_.classes_

        # store the loss, penalty and constraint configs if they were provided
        if configs is not None:
            self.fit_loss_ = configs['loss']
            self.fit_penalty_ = configs['penalty']
            self.fit_constraint_ = configs['constraint']

        # store optionization info if it was provided
        if opt_info is not None:
            self.opt_info_ = opt_info

        # store initial estimator if one was provided
        init_est = pre_pro_out.get('init_est', None)
        if init_est is not None:
            self.est_init_ = init_est

        return self

    def _set_fit_from(self, estimator, copy=False):
        """
        Sets the fit from a previously fit estimator i.e. gets all attributes ending in _

        Parameters
        ----------
        estimator:
            The fitted estimator whose fit attributes we want to get.
        copy: bool
            Whether or not to copy over the attributes.

        Output
        ------
        self
        """

        # TODO: what's the deal with starts with
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/utils/validation.py#L1095
        # v.endswith("_") and not v.startswith("__")

        for k, v in estimator.__dict__.items():
            if k.endswith('_'):

                if copy:
                    self.__dict__[k] = deepcopy(v)
                else:
                    self.__dict__[k] = v

        return self

    def decision_function(self, X):
        """
        The GLM decision function i.e. z = X.T @ coef + interept

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        Output
        ------
        z: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The decision function values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # TODO: for multi-response our coef_ is the transpose of sklearn's
        # convention. I think our choice of (n_features, n_responses)
        # Do we want to be stick with this choice?
        z = safe_sparse_dot(X, self.coef_,  # .T
                            dense_output=True)

        if hasattr(self, 'intercept_') and self.intercept_ is not None:
            z += self.intercept_

        return z

    def _more_tags(self):
        return {'requires_y': True}

    #############
    # Inference #
    #############

    def run_prefit_inference(self, X, y, sample_weight=None):
        """
        Runs statistical inference procedures on before fitting the model e.g. estimating the exponential family scale parameter.

        Parameters
        ----------
        X, y: array-like
            The training data used to fit the model.

        Output
        ------
        inferencer
        """
        if self.inferencer is not None:
            # TODO: do we want to do a copy here?
            inferencer = deepcopy(self.inferencer)
            inferencer.pre_fit(estimator=self, X=X, y=y,
                               sample_weight=sample_weight)
            return inferencer

        else:
            return None

    def run_postfit_inference(self, X, y, sample_weight=None):
        """
        Runs statistical inference procedures on the fitted model e.g. estimates the number of degrees of freedom. The inferencer_ attribute must first be set.

        Parameters
        ----------
        X, y: array-like
            The training data used to fit the model.

        Output
        ------
        self
        """

        # TODO-THINK-THOUGH: where do we want to set inferencer_?
        # right now this assumes inferencer_ has already been set in fit()
        # but this may be a bit unexpected
        if self.inferencer_ is not None:
            self.inferencer_.post_fit(estimator=self, X=X, y=y,
                                      sample_weight=sample_weight)

        return self

    ################################
    # sub-classes should implement #
    ################################

    # this is set by the LossMixin
    def _process_y(self, X, y, sample_weight=None, copy=True):
        """
        Processing for the y data e.g. transform class labels to indicator variables for multinomial.

        Parameters
        ---------
        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample

        copy: bool
            Whether or not to copy the X/y arrays or modify them in place.

        Output
        ------
        y: array-like
            The possibly transformed response data.
        """
        # subclass should overwrite
        raise NotImplementedError


class TunedGlm(BaseGlm):

    @autoassign
    def __init__(self,
                 loss='lin_reg',
                 penalty=None,
                 constraint=None,
                 standardize=True,
                 fit_intercept=True,
                 solver='default',
                 initializer='default',
                 inferencer=None,

                 select_metric=None,
                 scorer=None,
                 fit_eval=None,
                 verbose=0,
                 n_jobs=None,
                 pre_dispatch='2*n_jobs',
                 path_algo=True): pass

    @property
    def _is_tuner(self):
        return True

    def _prefit_and_setup_tuning(self, X, y, sample_weight=None):
        """
        TODO: document

        Parameters
        ----------

        Output
        ------
        tuner, X_pro, y_pro, sample_weight_pro, pre_pro_out,
        solver, solver_init, inferencer
        """

        # run any prefitting inference
        inferencer = self.run_prefit_inference(X, y, sample_weight)

        # preproceess X, y
        X_pro, y_pro, sample_weight_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        # possibly fit initializer estimator for flavored penalties
        #  that need one
        init_data_pro, init_est = \
            self.get_initializer(X=X, y=y,
                                 pre_pro_out=pre_pro_out,
                                 sample_weight=sample_weight)

        # store init_est here to be saved later in _pre_fit
        pre_pro_out['init_est'] = init_est

        # initialize config objects
        configs, solver, solver_init = \
            safe_initialize_configs(solver=self.solver,
                                    loss=self.loss,
                                    penalty=self.penalty,
                                    constraint=self.constraint,
                                    init_data=init_data_pro)

        ################
        # Setup tuning #
        ################

        # pull out flavor config from the penalty object
        if is_flavored(configs['penalty']):
            configs['flavor'] = safe_get_config(configs['penalty']).flavor

        # initialize tuning objects for each config
        for k in configs.keys():
            # if this is not already a tuner object, call the .tune() method
            if configs[k] is not None and  \
                    not isinstance(configs[k], TunerConfig):

                # create tuning object by default for all provided confings
                configs[k] = configs[k].tune()

        # setup tuning parameter grids from the data
        tuner = PenaltyPerLossFlavorTuner(**configs)
        tuner.set_tuning_values(X=X_pro, y=y_pro,
                                fit_intercept=self.fit_intercept,
                                sample_weight=sample_weight_pro)

        return tuner, X_pro, y_pro, sample_weight_pro, pre_pro_out,\
            solver, solver_init, inferencer

    def _get_select_metric(self):
        """
        Returns selection metric to use.

        Ouput
        -----
        select_metric: str
        """
        if self.select_metric is not None:
            return self.select_metric

        # if a multi-metric scorer was provide, use its default
        if self.scorer is not None and hasattr(self.scorer, 'default'):
            return self.scorer.default

        else:
            return 'score'

    def _run_fit_and_score_jobs(self, job_configs, store_ests=False):
        """
        Simply calls ya_glm.tune.backend.run_fit_and_score_jobs

        Parameters
        ----------
        jobs: iterable
            The job_configs argument to run_fit_and_score_jobs.

        store_ests: bool
            Whether or not to return the fit estimators.

        Output
        ------
        see run_fit_and_score_jobs()
        """

        return run_fit_and_score_jobs(job_configs=job_configs,
                                      store_ests=store_ests,
                                      scorer=self.scorer,
                                      fit_evals=self.fit_eval,
                                      n_jobs=self.n_jobs,
                                      verbose=self.verbose,
                                      pre_dispatch=self.pre_dispatch)


def safe_initialize_configs(solver, loss, penalty=None, constraint=None,
                            init_data=None):
    """
    Safely initializes the solver and configs where some of the configs might be ConfigTuner objects. Otherwise the same as initialize_configs().

    """

    configs = {'loss': loss, 'penalty': penalty, 'constraint': constraint}

    # pull out base configs from any tuning config that was provided
    tuners = {}
    keys = list(configs.keys())
    for k in keys:
        if isinstance(configs[k], TunerConfig):
            tuners[k] = configs.pop(k)

    # setup tuning configs
    configs_out, solver, init_values = \
        initialize_configs(solver=solver, init_data=init_data,
                           **configs,
                           **{k: v.base for (k, v) in tuners.items()})

    # put base initialized base config into tuning config
    for k, v in tuners.items():
        configs_out[k] = tuners[k].set_params(base=configs_out[k])

    return configs_out, solver, init_values


def initialize_configs(solver, loss, penalty=None, constraint=None,
                       init_data=None):
    """
    Initializers the solver and the loss, constraint, penalty, and penalty flavor configs. For example, this computes the adaptive weights from the initial coefficient for adaptive penalties.

    Parameters
    ----------
    solver: str, GlmSolver
        The desired Glm solver.

    loss: str, LossConfig
        The desired loss.

    penalty:  str, PenaltyConfig
        The desired penalty

    constraint: str, ConstraintConfig
        The desired constraint

    init_data: dict
        The preprocessed initializer data e.g. including the initializer coefficient.

    Output
    ------
    configs, solver, init_vals

    configs: dict
        The setup loss, constraint and penalty config objects. E.g. the penalty's adaptive weights are set here.

    solver: GlmSolver
        The configured Glm solver.

    solver_init: dict or None
        The values at which we should start the optimization problem.
    """
    assert type(penalty) != str, "TODO: need to add this"

    # optimization initialization
    # TODO-THINK-THROUGH: currently only used for NonConvexDirect
    # the logic of this code is a bit hard to follow at first glance
    solver_init = {}

    configs = {}
    configs['loss'] = get_loss_config(loss)
    configs['constraint'] = get_constraint_config(constraint)

    # possibly set initializers
    if is_flavored(penalty):
        configs['penalty'] = safe_get_config(penalty.flavor).\
            get_initialized_penalty(penalty=penalty,
                                    init_data=init_data)

        if safe_get_config(penalty.flavor).needs_separate_solver_init:
            solver_init = {'coef_init': penalty.coef_init_,
                           'intercept_init': penalty.intercept_init_}

    else:
        configs['penalty'] = penalty

    # get the GLM solver
    solver = get_solver(solver=solver,
                        **configs)

    # possibly get the LLA algorithm if needed
    solver = maybe_get_lla(penalty=configs['penalty'],
                           glm_solver=solver)

    return configs, solver, solver_init
