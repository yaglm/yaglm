from sklearn.base import is_classifier  # clone
from sklearn.model_selection._split import check_cv
import numpy as np
from copy import deepcopy
from numbers import Number
from time import time

from ya_glm.base import TunedGlm
from ya_glm.LossMixin import LossMixin

from ya_glm.tune.backend import get_cross_validation_jobs, \
    get_validation_jobs
from ya_glm.tune.select import select_tune_param, cv_select_tune_param

from ya_glm.autoassign import autoassign
from ya_glm.utils import get_from
from ya_glm.tune.utils import train_validation_idxs


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
                 initializer='default',
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

    def fit(self, X, y, sample_weight=None):

        start_time = time()
        tune_info = {'runtime': {}}

        ###########################
        # preprocessing and setup #
        ###########################

        # basic formatting check
        X, y, sample_weight = self._validate_data(X=X, y=y,
                                                  sample_weight=sample_weight)

        # prefitting routines: prefit inference, fit initializer,
        # pre process data, setup tuning parameter grid
        tuner, X_pro, y_pro, sample_weight_pro, pre_pro_out, \
            solver, solver_init, inferencer =\
            self._prefit_and_setup_tuning(X, y, sample_weight=sample_weight)

        self.tuner_ = tuner
        self.inferencer_ = inferencer

        tune_info['runtime']['prefit'] = time() - start_time

        ########################
        # run cross-validation #
        ########################

        start_time = time()

        # create CV folds
        cv = check_cv(cv=self.cv, y=y, classifier=is_classifier(self))

        # setup generator iterating over all the folds + parameter settings
        job_configs =\
            get_cross_validation_jobs(X=X, y=y,
                                      fold_iter=cv.split(X=X, y=y),
                                      est=self,
                                      solver=solver,
                                      tune_iter=self.tuner_,
                                      sample_weight=sample_weight,
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

        self._fit_from_configs(configs=best_tune_configs,
                               solver=solver,
                               pre_pro_out=pre_pro_out,
                               solver_init=solver_init,
                               X=X, y=y, X_pro=X_pro, y_pro=y_pro,
                               sample_weight=sample_weight,
                               sample_weight_pro=sample_weight_pro)

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
                 initializer='default',
                 inferencer=None,

                 val=0.2,

                 select_metric=None,
                 scorer=None,
                 fit_eval=None,
                 verbose=0,
                 n_jobs=None,
                 pre_dispatch='2*n_jobs',
                 path_algo=True): pass

    def fit(self, X, y, sample_weight=None):

        start_time = time()
        tune_info = {'runtime': {}}

        ###########################
        # preprocessing and setup #
        ###########################

        # basic formatting check
        X, y, sample_weight = self._validate_data(X=X, y=y,
                                                  sample_weight=sample_weight)

        # prefitting routines: prefit inference, fit initializer,
        # pre process data, setup tuning parameter grid
        tuner, X_pro, y_pro, sample_weight_pro, pre_pro_out, \
            solver, solver_init, inferencer =\
            self._prefit_and_setup_tuning(X, y, sample_weight=sample_weight)

        self.inferencer_ = inferencer
        self.tuner_ = tuner
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

        # setup generator iterating over all the folds + parameter settings
        job_configs =\
            get_validation_jobs(X=X, y=y,
                                est=self,
                                solver=solver,
                                tune_iter=self.tuner_,
                                train=train,
                                test=test,
                                sample_weight=sample_weight,
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

        self._fit_from_configs(configs=best_tune_configs,
                               solver=solver,
                               pre_pro_out=pre_pro_out,
                               solver_init=solver_init,
                               X=X, y=y, X_pro=X_pro, y_pro=y_pro,
                               sample_weight=sample_weight,
                               sample_weight_pro=sample_weight_pro)

        tune_info['runtime']['refit'] = time() - start_time

        self.tune_info_ = tune_info
        return self


class GlmTrainMetric(LossMixin, TunedGlm):
    """
    Tunes a GLM model using generalized cross-validation (GCV).
    """

    def fit(self, X, y, sample_weight=None):

        tune_info = {'runtime': {}}
        start_time = time()

        ###########################
        # preprocessing and setup #
        ###########################

        # basic formatting check
        X, y, sample_weight = self._validate_data(X=X, y=y,
                                                  sample_weight=sample_weight)
        # prefitting routines: prefit inference, fit initializer,
        # pre process data, setup tuning parameter grid
        tuner, X_pro, y_pro, sample_weight_pro, pre_pro_out, \
            solver, solver_init, inferencer =\
            self._prefit_and_setup_tuning(X, y, sample_weight=sample_weight)

        self.tuner_ = tuner
        self.inferencer_ = inferencer

        tune_info['runtime']['prefit'] = time() - start_time

        #######################
        # Compute tuning path #
        #######################

        start_time = time()

        # setup generator iterating over all the folds + parameter settings
        job_configs =\
            get_validation_jobs(X=X, y=y,
                                est=self,
                                solver=solver,
                                tune_iter=self.tuner_,
                                train=None, test=None,  # training only!
                                sample_weight=sample_weight,
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
