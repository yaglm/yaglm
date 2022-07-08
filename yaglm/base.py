import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, _check_y, \
    _check_sample_weight, FLOAT_DTYPES

from yaglm.autoassign import autoassign
from yaglm.processing import process_X, deprocess_fit, process_init_data, \
    _check_offsets
from yaglm.utils import fit_if_unfitted, get_coef_and_intercept, \
    is_str_and_matches, get_shapes_from

from yaglm.config.loss import get_loss_config
from yaglm.config.constraint import get_constraint_config
from yaglm.config.penalty import get_penalty_config
from yaglm.config.penalty_utils import get_flavor_kind, get_unflavored
from yaglm.config.base_params import get_base_config
from yaglm.solver.default import get_solver
from yaglm.solver.LLA import LLAFixedInit

from yaglm.tune.backend import run_fit_and_score_jobs
from yaglm.tune.combined_tuner import PenaltyPerLossFlavorTuner


class BaseGlm(BaseEstimator):
    """
    Base class for GLMs.

    Parameters
    ----------
    loss: str, LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See yaglm.LossConfig for available loss functions.

    penalty: None, PenaltyConfig
        The penalty config object specifying the penalty e.g. Lasso, Ridge, ...

    constraint: None, ConstraintConfig
        (Optional) The constraint config object e.g. Positive, Isotonic, ...

    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Standardization means mean centering and scaling each column by its standard deviation. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    solver: str, yaglm.base.GlmSolver
        The solver used to solve the penalized GLM optimization problem. If this is set to 'default' we try to guess the best solver. Otherwise a custom solver can be provided by specifying a GlmSolver object.

    lla: bool, LLASolver
        Whether or not to use the LLA algorithm for non-convex penalties. If False, will attempt to solve the non-convex problem directly using the solver. If True, the LLA algorithm will be used for non-convex problems using the solver for the subproblems. Note the default LLA algorithm only takes one LLA step. An LLA algorithm object can be provided that specifies the LLA algorithm's behavior.

    initializer: str, dict, Estimator
        Specifies the initial estimator to use for penalties that require an initial estimator e.g. adaptive and non-convex LLA penalties. If str, must be one of ['default', 'zero']. If init='default', will infer a reasonable default penalty to use. If 'init'='zero', will initialize from zero. If an estimator is provided, will fit the estimator to the training data (unless it is already fit) then use the estimtor's fit coefficient. If a dict is provided, it must include the key 'coef' for the initializer coefficient.

    relaxed: bool
        Fit the relaxed version of the penalty i.e. fit the penalty, obtain the estimated support set, then fit the unpenalized version of the problem only including the estimated support.

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
                 lla=True,
                 initializer='default',
                 relaxed=False,
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
        loss_config = get_base_config(get_loss_config(self.loss))
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
        raise NotImplementedError("Subclass should overwrite")

    def _validate_data(self, X, y, sample_weight=None, offsets=None,
                       accept_sparse=True):
        """
        Validates the X/y data. This should not change the raw input data, but may reformat the data (e.g. convert pandas to numpy).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None, array-like, shape (n_samples, )
            (Optional) The sample weights

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        accept_sparse: bool
            Whether or not X is allowed to be sparse.

        Output
        ------
        X, y, sample_weight, offsets
        """
        X = check_array(X, accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        offsets = _check_offsets(offsets, X=X, dtype=X.dtype,
                                 force_not_none=False,
                                 force_vector=False)

        y = _check_y(y, multi_output=True, y_numeric=False)

        # make sure 1d input is actually a vector
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        # make sure X, y have same number of samples
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows!")

        return X, y, sample_weight, offsets

    def preprocess(self, X, y, sample_weight=None, offsets=None,
                   copy=True, check_input=True):
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

        sample_weight: None, array-like, shape (n_samples, )
            (Optional) The sample weights.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        copy: bool
            Whether or not to copy the X/y arrays or modify them in place.

        Output
        ------
        pro_data, pre_pro_out

        pro_data: dict
            The processed data. Has keys

            X: array-like, shape (n_samples, n_features)
                The possibly transformed covariate data.

            y: array-like, shape (n_samples, )
                The possibly transformed response data.

            sample_weight: None or array-like,  shape (n_samples,)
                The processed sample weights. Ensures sum(sample_weight) = n_samples. Possibly incorporate class weights.

        pro_pro_out: dict
            Data from preprocessing e.g. X_center, X_scale.
        """

        if sample_weight is not None:
            if copy:
                sample_weight = sample_weight.copy()

        if offsets is not None:
            if hasattr(offsets, 'copy'):
                offsets = offsets.copy()
            else:
                offsets = deepcopy(offsets)

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

        out.update(y_out)

        pro_data = {'X': X, 'y': y,
                    'sample_weight': sample_weight, 'offsets': offsets}
        return pro_data, out

    def get_unflavored_tunable(self):
        """
        Returns an unflavored and tunable version of this estimator. If this estimator is not tunable, will return the cross-validation version by default.

        This should copy all data.
        """
        raise NotImplementedError("Subclass should overwrite")

    def get_initializer(self, X, y, pre_pro_out,
                        sample_weight=None, offsets=None):
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

        offsets: None, float, array-like, shape (n_samples, )
            The offsets for each sample.

        Output
        ------
        init_data, init_est

        init_data: None, dict
            The initialization data.

        init_est: None, Estimator
            The initializer estimator if one was fit.
        """

        ######################################
        # Initializer for flavored penalties #
        ######################################

        flavor_kind = get_flavor_kind(self.penalty)

        # flavored penalties may require fitting an initializer
        if flavor_kind is not None:

            ##############################################
            # Get initializer coefficient/intercept data #
            ##############################################

            if is_str_and_matches(self.initializer, 'default'):

                # by default adaptive and non-convex LLA algorithms
                # use the unflavored estimator as a default initializer
                if flavor_kind == 'adaptive' or \
                        (flavor_kind in ['non_convex', 'mixed'] and self.lla):
                    init_est = self.get_unflavored_tunable()
                    yes_pro_pro_init = True
                else:
                    init_data = None
                    init_est = None

            # initialize at zero
            elif is_str_and_matches(self.initializer, 'zero'):
                coef_shape, intercept_shape = get_shapes_from(X=X, y=y)
                coef = np.zeros(coef_shape)
                if len(intercept_shape) == 0:
                    intercept = 0
                else:
                    intercept = np.zeros(intercept_shape)

                init_data = {'coef': coef, 'intercept': intercept}
                yes_pro_pro_init = False
                init_est = None

            # if initializer is a dict, just return initializer
            elif isinstance(self.initializer, dict):
                init_data = self.initializer

                # check whether or not we should apply the preprocessing
                # to the init data
                # TODO: document this!
                yes_pro_pro_init = init_data.get('pre_pro', True)
                init_est = None

            # user provided an initial estimator
            else:
                init_est = self.initializer
                yes_pro_pro_init = True

            # Possibly fit initial estimator
            if init_est is not None:
                # possibly fit this estimator if it has not already been fit
                init_est = fit_if_unfitted(init_est, X=X, y=y,
                                           sample_weight=sample_weight,
                                           offsets=offsets)

                coef, intercept = get_coef_and_intercept(init_est, copy=True,
                                                         error=True)

                init_data = {'coef': coef, 'intercept': intercept}

            # possibly preprocess the init data
            if init_data is not None and yes_pro_pro_init:
                init_data = \
                    process_init_data(init_data=init_data,
                                      pre_pro_out=pre_pro_out)

            # adaptive Lasso may need to know the number of training samples
            if flavor_kind in ['adaptive', 'mixed']:
                init_data['n_samples'] = X.shape[0]

            return init_data, init_est

        else:
            init_data, init_est = None, None
            return init_data, init_est

    def setup_and_prefit(self, X, y, sample_weight, offsets):
        """
        Runs various routines needed before fitting the GLM

        - validate input data
        - (possibly) run prefit inference (e.g. estimate scale parameter)
        - preprocesses raw data (e.g. center/scale)
        - (possibly) get initializer e.g. an initial estimator for adaptive/non-convex penalties
        - setup loss, penalty, and constraint configs
        - setup the solver. Note the LLA initializer is fixed here.


        Note the adaptive weights are not computed here!

        This method does not set any attributes.

        Output
        ------
        pro_data, raw_data, pre_pro_out,
            configs, solver, init_data, inferencer

        pro_data: dict
            The processed data; has keys ['X', 'y', 'sample_weight']

        raw_data: dict
            The raw data; has keys ['X', 'y', 'sample_weight']

        pre_pro_out: dict
            The preprocessing output.

        configs: dict
            The loss, penalty and constraint configs; has keys ['loss', 'penalty', 'constraint'].

        solver: GlmSolver
            The initialized solver object.

        init_data: dict
            The initialization data.

        inferencer: None, Inferencer
            The inferencer object. Note the prefitting inference has been run.
        """

        #################
        # preprocessing #
        #################

        # basic formatting check
        X, y, sample_weight, offsets = \
            self._validate_data(X=X, y=y,
                                sample_weight=sample_weight,
                                offsets=offsets)

        raw_data = {'X': X, 'y': y,
                    'sample_weight': sample_weight, 'offsets': offsets}

        # run any prefitting inference
        inferencer = self.run_prefit_inference(**raw_data)

        # preproceess X, y
        pro_data, pre_pro_out = self.preprocess(**raw_data, copy=True)

        # possibly fit initializer estimator for flavored penalties
        # that need one
        init_data, init_est = \
            self.get_initializer(X=X, y=y,
                                 pre_pro_out=pre_pro_out,
                                 sample_weight=sample_weight,
                                 offsets=offsets)

        # store init_est here to be saved later in _pre_fit
        pre_pro_out['init_est'] = init_est

        #################
        # setup configs #
        #################

        # setup loss, constraint and penalty configs
        # TODO: copy or clone?
        configs = {'loss': get_loss_config(self.loss),
                   'constraint': get_constraint_config(self.constraint),
                   'penalty': get_penalty_config(self.penalty)
                   }
        # TODO: somewhere run get_flavor_config to setup penalty flavor from string

        ################
        # setup solver #
        ################
        flavor_kind = get_flavor_kind(configs['penalty'])

        if flavor_kind in ['non_convex', 'mixed'] and self.lla:
            # LLA algorithm

            # default LLA solver
            if type(self.lla) == bool:
                # TODO: do we want max_steps=1 by default?
                solver = LLAFixedInit(max_steps=1)
            else:
                solver = self.lla  # TODO: should there be a copy or clone?

            # set subproblem solver
            solver.set_sp_solver(get_solver(self.solver, **configs))

        else:
            # user specified solver!
            solver = get_solver(self.solver, **configs)

        # possibly set fixed initialization e.g. for the LLA algorithm
        if solver.needs_fixed_init:
            solver.set_fixed_init(init_data)

        return pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer

    def _get_solver_init(self, init_data):
        flavor_kind = get_flavor_kind(self.penalty)

        if flavor_kind in ['non_convex', 'mixed'] and not self.lla:
            # for non-convex direct we are allowed to specify the
            # initial value for optimization.
            # for the LLA algorithm this is specified somewhere else

            if init_data is None:
                return {}

            else:
                return {'coef_init': init_data.get('coef', None),
                        'intercept_init': init_data.get('intercept', None)}
        else:
            return None

    def _fit_from_configs(self, pro_data, raw_data, configs, solver,
                          pre_pro_out, init_data=None):
        """
        Fits the specified GLM. Possibly runs after fitting inference.

        Parameters
        ----------
        configs: dict of ParamConfigs
            The configs that specify the loss, penalty and constraint. Note these should be ParamConfig, not TunerConfigs.

        see the output of self.setup_and_prefit()

        Output
        ------
        self
        """

        # set the solver initialization data
        # only used for non-convex, non-lla algorithm
        solver_init = self._get_solver_init(init_data)

        ##########
        # solve! #
        ##########

        # setup solver
        solver.setup(fit_intercept=self.fit_intercept,
                     **pro_data,  # X, y, sample_weight, offsets
                     **configs,  # loss, penalty, constraint
                     )

        solver_init = {} if solver_init is None else solver_init
        fit_out, _,  opt_info = solver.solve(**solver_init)

        ###############
        # Fit relaxed #
        ###############
        if self.relaxed:
            raise NotImplementedError("TODO add")
            # get unpenalized version of the penalty
            # find support of estimated coefficient
            # setup and solve relaxed problem
            # map relaxed solution back to full vector
            # TODO: give option to provide different solver for relaxed

        ##################
        # post procesing #
        ##################

        # set the fit coefficient e.g. undo preprocessing scaling
        self._set_fit(fit_out=fit_out,
                      pre_pro_out=pre_pro_out,
                      configs=configs,
                      opt_info=opt_info)

        # run any after fitting statistical inference
        # e.g. estimate number of DoF
        self.run_after_fit_inference(**raw_data)  # X, y, sample_weight

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

    def decision_function(self, X, offsets=None):
        """
        The GLM decision function i.e. z = X.T @ coef + interept or

        z = X.T @ coef + interept + offests

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        offsets: None, array-like, shape (n_samples, )
            (Optional) Offsets for the decision function.
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
        z = safe_sparse_dot(X, self.coef_, dense_output=True)

        if hasattr(self, 'intercept_') and self.intercept_ is not None:
            z += self.intercept_

        if offsets is not None:
            z += offsets

        return z

    def _more_tags(self):
        return {'requires_y': True}

    #############
    # Inference #
    #############

    def run_prefit_inference(self, X, y, sample_weight=None, offsets=None):
        """
        Runs statistical inference procedures on before fitting the model e.g. estimating the exponential family scale parameter.

        Parameters
        ----------
        X, y: array-like
            The training data used to fit the model.

        sample_weight: None, array-like, shape (n_samples, )
            (Optional) The sample weights

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        inferencer
        """
        if self.inferencer is not None:
            # TODO: do we want to do a copy here?
            inferencer = deepcopy(self.inferencer)
            inferencer.pre_fit(estimator=self, X=X, y=y,
                               sample_weight=sample_weight,
                               offsets=offsets)
            return inferencer

        else:
            return None

    def run_after_fit_inference(self, X, y, sample_weight=None, offsets=None):
        """
        Runs statistical inference procedures on the fitted model e.g. estimates the number of degrees of freedom. The inferencer_ attribute must first be set.

        Parameters
        ----------
        X, y: array-like
            The training data used to fit the model.

        sample_weight: None, array-like, shape (n_samples, )
            (Optional) The sample weights

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        self
        """

        # TODO-THINK-THOUGH: where do we want to set inferencer_?
        # right now this assumes inferencer_ has already been set in fit()
        # but this may be a bit unexpected
        if self.inferencer_ is not None:
            self.inferencer_.after_fit(estimator=self, X=X, y=y,
                                       sample_weight=sample_weight,
                                       offsets=offsets)

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
                 lla=True,
                 initializer='default',
                 relaxed=False,
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

    def get_tuner(self, configs, pro_data, init_data):
        """
        Creates a tuner object for tuning over the loss, constraints, penalty and penalty flavors. Note the adaptive weights are also set here.

        Parameters
        ----------
        configs: dict
            A dict of the loss, penalty, and constraint configs.

        pro_data: dict
            The preprocessed X, y, sample_weight, and offset data.

        init_data: dict
            The initialization data for the solver. This i

        Output
        ------
        tuner: PenaltyPerLossFlavorTuner
            The tuner object with set_tuning_values() already called.
        """

        #
        init_data = {} if init_data is None else init_data
        init_data['lla'] = self.lla  # tell the init data if
        # we are using the lla algorithm
        # TODO-THINK-THROUGH: is this a bit of a cluncky place to
        # say we are using the LLA algorith?

        # setup tuning parameter grids from the data
        tuner = PenaltyPerLossFlavorTuner(**configs)

        # this creates all the tuning parameter grids
        # the adaptive weights are set in this call too
        # since they may vary if we tune over the adative expon
        tuner.set_tuning_values(fit_intercept=self.fit_intercept,
                                init_data=init_data,
                                **pro_data)
        return tuner

    def get_tune_param_seq(self):
        """
        Output
        ------
        tune_seq: pd.DataFrame of list of dicts
            The tuning parameter settings.
        """
        if not hasattr(self, 'tune_results_'):
            raise RuntimeError("No tuning parameter grid detected")

        return self.tune_results_['params']

    def get_unflavored_tunable(self):
        """
        Gets an unflavored version of this estimator.

        Output
        ------
        est: Estimator
            The same estimator but ensures the penalty is not flavored. Note this copies all the parameters of the estimator.
        """

        # TODO: possibly avoid copying the initializer?
        est = deepcopy(self)

        # set unflavored penalty
        unflavored_penalty = get_unflavored(est.penalty)
        est.set_params(penalty=unflavored_penalty)

        est.set_params(initializer='default')  # no need for init

        return est

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
        Simply calls yaglm.tune.backend.run_fit_and_score_jobs

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
        if self.relaxed:
            raise NotImplementedError("TODO: add")

        return run_fit_and_score_jobs(job_configs=job_configs,
                                      store_ests=store_ests,
                                      scorer=self.scorer,
                                      fit_evals=self.fit_eval,
                                      relaxed=self.relaxed,
                                      n_jobs=self.n_jobs,
                                      verbose=self.verbose,
                                      pre_dispatch=self.pre_dispatch)
