from sklearn.base import is_classifier  # clone
from sklearn.model_selection._split import check_cv
import numpy as np
from copy import deepcopy
from numbers import Number
from time import time

from yaglm.base import TunedGlm
from yaglm.LossMixin import LossMixin

from yaglm.tune.backend import get_cross_validation_jobs, \
    get_validation_jobs, get_train_jobs
from yaglm.tune.select import select_tune_param, cv_select_tune_param

from yaglm.autoassign import autoassign
from yaglm.utils import get_from
from yaglm.tune.utils import train_validation_idxs


class GlmCV(LossMixin, TunedGlm):
    """
    Tunes a GLM model using cross-validation.
    """

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

                 cv=None,
                 select_rule='best',

                 select_metric=None,
                 scorer=None,
                 fit_eval=None,
                 verbose=0,
                 n_jobs=None,
                 pre_dispatch='2*n_jobs',
                 path_algo=True): pass

    def fit(self, X, y, sample_weight=None, offsets=None):
        """
        Fits the penalized GLM and tunes the parameters with cross-validation.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: None or array-like, shape (n_samples,)
            (Optional) Individual weights for each sample.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        self
            Fitted estimator.
        """

        start_time = time()
        tune_info = {'runtime': {}}

        ##############################################
        # setup, preprocess, and prefitting routines #
        ##############################################
        pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer = \
            self.setup_and_prefit(X=X, y=y,
                                  sample_weight=sample_weight,
                                  offsets=offsets)

        # store inferencer
        self.inferencer_ = inferencer

        ###############################
        # setup tuning parameter grid #
        ###############################

        # setup tuning parameter grids from the data
        self.tuner_ = self.get_tuner(configs=configs,
                                     pro_data=pro_data,
                                     init_data=init_data)

        tune_info['runtime']['prefit'] = time() - start_time

        ########################
        # run cross-validation #
        ########################
        start_time = time()

        # create CV folds
        cv = check_cv(cv=self.cv, y=y, classifier=is_classifier(self))

        # set the solver initialization data
        # only used for non-convex, non-lla algorithm
        solver_init = self._get_solver_init(init_data)

        # setup generator iterating over all the folds + parameter settings
        job_configs =\
            get_cross_validation_jobs(raw_data=raw_data,
                                      fold_iter=cv.split(X=X, y=y),
                                      est=self,
                                      solver=solver,
                                      tune_iter=self.tuner_,
                                      path_algo=self.path_algo,
                                      solver_init=deepcopy(solver_init)
                                      )

        # fit and score all models!
        self.tune_results_ = self._run_fit_and_score_jobs(job_configs)

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            cv_select_tune_param(self.tune_results_,
                                 metric=self._get_select_metric(),
                                 rule=self.select_rule,
                                 prefer_larger_param=True)

        # double check we used 1se rule correctly
        # TODO: get 1se rule working more generally!
        if self.select_rule == '1se':
            p = list(self.tune_results_['params'][0].keys())

            if len(p) > 1 or p[0] != 'penalty__pen_val':
                raise NotImplementedError("1se rule is not currently implemented for multi-parameter tuning or for tuning anything but penalty__pen_val")

        tune_info['runtime']['tune'] = time() - start_time

        ##########################################
        # solve with the best parameter settings #
        ##########################################
        start_time = time()

        best_tune_configs = get_from(self.tuner_.iter_configs(),
                                     idx=self.best_tune_idx_)

        self._fit_from_configs(pro_data=pro_data, raw_data=raw_data,
                               configs=best_tune_configs,
                               solver=solver,
                               pre_pro_out=pre_pro_out,
                               init_data=init_data)

        tune_info['runtime']['refit'] = time() - start_time
        self.tune_info_ = tune_info
        return self


class GlmValidation(LossMixin, TunedGlm):
    """
    Tunes a GLM model using a validation set.
    """

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

                 val=0.2,

                 select_metric=None,
                 scorer=None,
                 fit_eval=None,
                 verbose=0,
                 n_jobs=None,
                 pre_dispatch='2*n_jobs',
                 path_algo=True): pass

    def fit(self, X, y, sample_weight=None, offsets=None):
        """
        Fits the penalized GLM and tunes the parameters with a validation set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: None or array-like, shape (n_samples,)
            (Optional) Individual weights for each sample.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        self
            Fitted estimator.
        """
        start_time = time()
        tune_info = {'runtime': {}}

        ##############################################
        # setup, preprocess, and prefitting routines #
        ##############################################
        pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer = \
            self.setup_and_prefit(X=X, y=y,
                                  sample_weight=sample_weight,
                                  offsets=offsets)

        # store inferencer
        self.inferencer_ = inferencer

        ###############################
        # setup tuning parameter grid #
        ###############################

        # setup tuning parameter grids from the data
        self.tuner_ = self.get_tuner(configs=configs,
                                     pro_data=pro_data,
                                     init_data=init_data)

        tune_info['runtime']['prefit'] = time() - start_time

        ##################
        # run validation #
        ##################
        start_time = time()

        # do train/test split
        n_samples = X.shape[0]
        if isinstance(self.val, Number):
            train, test = train_validation_idxs(n_samples=X.shape[0],
                                                test_size=self.val,
                                                shuffle=True,
                                                random_state=None,
                                                y=y,
                                                classifier=is_classifier(self))

        else:
            # user provided test set
            test = np.array(self.val)
            train = np.array(list(set(range(n_samples)).difference(test)))

        # set the solver initialization data
        # only used for non-convex, non-lla algorithm
        solver_init = self._get_solver_init(init_data)

        # setup generator iterating over all the folds + parameter settings
        job_configs =\
            get_validation_jobs(raw_data=raw_data,
                                est=self,
                                solver=solver,
                                tune_iter=self.tuner_,
                                train=train,
                                test=test,
                                path_algo=self.path_algo,
                                solver_init=deepcopy(solver_init)
                                )

        # fit and score all models
        self.tune_results_, _ = self._run_fit_and_score_jobs(job_configs)

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            select_tune_param(self.tune_results_,
                              kind='test',
                              metric=self._get_select_metric())
        tune_info['runtime']['tune'] = time() - start_time

        ##########################################
        # solve with the best parameter settings #
        ##########################################
        start_time = time()

        best_tune_configs = get_from(self.tuner_.iter_configs(),
                                     idx=self.best_tune_idx_)

        self._fit_from_configs(pro_data=pro_data, raw_data=raw_data,
                               configs=best_tune_configs,
                               solver=solver,
                               pre_pro_out=pre_pro_out,
                               init_data=init_data)

        tune_info['runtime']['refit'] = time() - start_time

        self.tune_info_ = tune_info
        return self


class GlmTrainMetric(LossMixin, TunedGlm):
    """
    Tunes a GLM model using generalized cross-validation (GCV).
    """

    def fit(self, X, y, sample_weight=None, offsets=None):
        """
        Fits the penalized GLM and tunes the parameters with a metric from the training data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: None or array-like, shape (n_samples,)
            (Optional) Individual weights for each sample.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        self
            Fitted estimator.
        """

        tune_info = {'runtime': {}}
        start_time = time()

        ##############################################
        # setup, preprocess, and prefitting routines #
        ##############################################
        pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer = \
            self.setup_and_prefit(X=X, y=y,
                                  sample_weight=sample_weight,
                                  offsets=offsets)

        # store inferencer
        self.inferencer_ = inferencer

        ###############################
        # setup tuning parameter grid #
        ###############################

        # setup tuning parameter grids from the data
        self.tuner_ = self.get_tuner(configs=configs,
                                     pro_data=pro_data,
                                     init_data=init_data)

        tune_info['runtime']['prefit'] = time() - start_time

        #######################
        # Compute tuning path #
        #######################

        start_time = time()

        # set the solver initialization data
        # only used for non-convex, non-lla algorithm
        solver_init = self._get_solver_init(init_data)

        # setup generator iterating over all the folds + parameter settings
        job_configs = get_train_jobs(pro_data=pro_data,
                                     raw_data=raw_data,
                                     pre_pro_out=pre_pro_out,
                                     est=self,
                                     solver=solver,
                                     tune_iter=self.tuner_,
                                     path_algo=self.path_algo,
                                     solver_init=deepcopy(solver_init)
                                     )

        # fit and score all models!
        self.tune_results_, estimators = \
            self._run_fit_and_score_jobs(job_configs, store_ests=True)

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            select_tune_param(self.tune_results_,
                              kind='train',
                              metric=self._get_select_metric())

        # TODO: perhasps give option to save tuning path
        # store all fit coefficeints/intercepts
        # self.coef_path_ = [e.coef_ for e in estimators]
        # if self.fit_intercept:
        #     self.intercept_path_ = [e.intercept_ for e in estimators]
        tune_info['runtime']['tune'] = time() - start_time

        ###############################################
        # select fit with the best parameter settings #
        ###############################################
        start_time = time()

        self._set_fit_from(estimators[self.best_tune_idx_])

        tune_info['runtime']['refit'] = time() - start_time
        self.tune_info_ = tune_info

        return self
