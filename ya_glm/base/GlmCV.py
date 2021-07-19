from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import check_cv
from functools import partial

from time import time
from textwrap import dedent

from ya_glm.autoassign import autoassign
from ya_glm.init_signature import add_from_classes
from ya_glm.make_docs import merge_param_docs
from ya_glm.pen_seq import get_pen_val_seq
from ya_glm.cv.cv_select import CVSlectMixin  # select_best_cv_tune_param
from ya_glm.cv.run_cv import run_cv_grid, run_cv_path, \
    add_params_to_cv_results, score_from_fit_path


# TODO: move estimator descripting to subclasses
_cv_params = dedent("""
estimator: estimator object
    The base estimator to be cross-validated.

cv: int, cross-validation generator or an iterable, default=None
    Determines the cross-validation splitting strategy.

cv_select_metric: None, str
    Which metric to use for select the best tuning parameter if multiple metrics are computed.

cv_scorer: None, callable(est, X, y) -> dict or float
    A function for evaluating the cross-validation fit estimators.
    If this returns a dict of multiple scores then cv_select_metric determines which metric is used to select the tuning parameter.

cv_n_jobs: None, int
    Number of jobs to run in parallel.

cv_verbose: int
    Amount of printout during cross-validation.

cv_pre_dispatch: int, or str, default=n_jobs
    Controls the number of jobs that get dispatched during parallel execution
""")


class GlmCV(CVSlectMixin, BaseEstimator):
    """
    Base class for generalized linear models tuned with cross-validation.
    """

    # subclass may implement the following

    # description
    _descr = None

    # penalty parameter description
    _params_descr = _cv_params

    _attr_descr = dedent("""
        best_estimator_:
            The fit estimator with the parameters selected via cross-validation.

        cv_results_: dict
            The cross-validation results.

        best_tune_idx_: int
            Index of the best tuning parameter. Indexes the list returned by get_tuning_sequence().

        best_tune_params_: dict
            The best tuning parameters.

        cv_data_: dict
            Additional data about the CV fit e.g. the runtime.
        """)

    @autoassign
    def __init__(self,
                 estimator,

                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs'):
        pass

    def fit(self, X, y, sample_weight=None):
        """
        Runs cross-validation then refits the GLM with the selected tuning parameter.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.
        """

        # check the input data
        self._check_base_estimator(self.estimator)
        est = clone(self.estimator)
        X, y, sample_weight = est._validate_data(X, y,
                                                 sample_weight=sample_weight)

        # set up the tuning parameter values using the processed data
        self._set_tuning_values(X=X, y=y, sample_weight=sample_weight)

        # maybe add sample weight to fit params
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
        else:
            fit_params = None

        ########################
        # run cross-validation #
        ########################
        start_time = time()
        if self.has_path_algo:
            # if we have a path algorithm available use it

            self.cv_results_ = \
                self._run_cv_path(estimator=est, X=X, y=y, cv=self.cv,
                                  fit_params=fit_params)

        else:
            # otherwise just do grid search
            self.cv_results_ = \
                self._run_cv_grid(estimator=est, X=X, y=y, cv=self.cv,
                                  fit_params=fit_params)

        self.cv_data_ = {'cv_runtime':  time() - start_time}

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)

        # set best tuning params
        est.set_params(**self.best_tune_params_)

        #########################################
        # refit with selected parameter setting #
        #########################################
        start_time = time()
        self.best_estimator_ = est.fit(X, y, sample_weight=sample_weight)
        self.cv_data_['refit_runtime'] = time() - start_time

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        check_is_fitted(self)
        return self.best_estimator_.score(X, y)

    def decision_function(self, X):
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        if not hasattr(self.best_estimator_, 'predict_proba'):
            raise NotImplementedError("This method does not have a preidct_proba function")
        else:
            return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        check_is_fitted(self)
        if not hasattr(self.best_estimator_, 'predict_log_proba'):
            raise NotImplementedError("This method does not have a predict_log_proba function")
        else:
            return self.best_estimator_.predict_log_proba(X)

    ####################
    # Cross-validation #
    ####################
    @property
    def has_path_algo(self):
        return hasattr(self.estimator, 'solve_glm_path') and \
            self.estimator.solve_glm_path is not None

    def _run_cv_grid(self, estimator, X, y=None,
                     cv=None, fit_params=None):

        cv_results = run_cv_grid(X, y,
                                 estimator=estimator,
                                 param_grid=self.get_tuning_param_grid(),
                                 cv=cv,
                                 scoring=self.cv_scorer,
                                 fit_params=fit_params,
                                 n_jobs=self.cv_n_jobs,
                                 verbose=self.cv_verbose,
                                 pre_dispatch=self.cv_pre_dispatch)

        return cv_results

    def _run_cv_path(self, estimator, X, y=None, cv=None, fit_params=None):

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        cv_results, _ = \
            run_cv_path(X=X, y=y,
                        fold_iter=cv.split(X, y),
                        fit_and_score_path=fit_and_score_path,
                        kws=self._get_solve_path_kws(),
                        fit_params=fit_params,
                        include_spilt_vals=True,  # TODO: maybe give option for this?
                        add_params=False,
                        n_jobs=self.cv_n_jobs,
                        verbose=self.cv_verbose,
                        pre_dispatch=self.cv_pre_dispatch)

        # add parameter sequence to CV results
        # this allows us to pass fit_path a one parameter sequence
        # while cv_results_ uses different names
        param_seq = self.get_tuning_sequence()
        cv_results = add_params_to_cv_results(param_seq=param_seq,
                                              cv_results=cv_results)

        return cv_results

    def _fit_and_score_path_getter(self, estimator):

        # get the correct fit and score method
        est = clone(estimator)
        def est_from_fit(fit_out, pre_pro_out):
            est._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
            return est

        if hasattr(est, 'preprocess'):
            preprocess = est.preprocess
        else:
            preprocess = None

        fit_and_score_path = partial(score_from_fit_path,
                                     solve_path=est.solve_glm_path,
                                     est_from_fit=est_from_fit,
                                     scorer=self.cv_scorer,
                                     preprocess=preprocess)

        return fit_and_score_path

    ################################
    # Subclasses need to implement #
    ################################
    def _get_solve_path_kws(self):
        """
        The solve path function will be call as

        solve_path(X=X, y=y, **kws)

        Output
        ------
        kws: dict
            All keyword arguments for computing the path.
        """
        raise NotImplementedError

    def check_base_estimator(self, estimator):
        """
        Check the base estimator aggrees with the CV class
        """
        raise NotImplementedError

    def _set_tuning_values(self, X, y, **kws):
        """
        Sets the tuning parameter sequence from the transformed data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The processed training covariate data.

        y: array-like, shape (n_samples, )
            The processed training response data.

        **kws:
            Additional keyword arguments.
        """
        # subclass should overwrite
        raise NotImplementedError

    def get_tuning_sequence(self):
        """
        Returns the tuning parameters in a sequence.
        """
        raise NotImplementedError


_pen_seq_params = dedent("""
n_pen_vals: int
    Number of penalty values to try for automatically generated tuning parameter sequence.

pen_vals: None, array-like
    (Optional) User provided penalty value sequence. The penalty sequence should be monotonicly decreasing so the homotopy path algorithm works propertly.

pen_min_mult: float
    Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

pen_spacing: str
    How the penalty values are spaced. Must be one of ['log', 'lin']
    for logarithmic and linear spacing respectively.
""")


class GlmCVSinglePen(GlmCV):
    """
    Base class for tuning a GLM with a single penaly parameter with cross-validation.
    """

    _param_descr = merge_param_docs(GlmCV._params_descr,
                                    _pen_seq_params)

    @add_from_classes(GlmCV, add_first=False)
    def __init__(self,
                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log'
                 ):
        pass

    def _set_tuning_values(self, X, y, sample_weight=None):
        if self.pen_vals is None:
            pen_val_max = self.estimator.\
                get_pen_val_max(X=X, y=y, sample_weight=sample_weight)
        else:
            pen_val_max = None

        self._set_tune_from_pen_max(pen_val_max=pen_val_max)

    def _set_tune_from_pen_max(self, pen_val_max=None):
        self.pen_val_seq_ = \
            get_pen_val_seq(pen_val_max,
                            n_pen_vals=self.n_pen_vals,
                            pen_vals=self.pen_vals,
                            pen_min_mult=self.pen_min_mult,
                            pen_spacing=self.pen_spacing)

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values.
        Make sure the method that computes the cross-validation results
        orders the parameters in the same order as get_tuning_sequence()
        Output
        ------
        values: iterable
        """
        return list(ParameterGrid(self.get_tuning_param_grid()))

    def get_tuning_param_grid(self):
        """
        Returns tuning parameter grid.

        Output
        ------
        param_grid: dict of lists
        """
        return {'pen_val': self.pen_val_seq_}
