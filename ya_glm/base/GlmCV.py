import numpy as np
from time import time
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid

from ya_glm.cv.cv_select import CVSlectMixin
from ya_glm.pen_seq import get_sequence_decr_max
from ya_glm.autoassign import autoassign


class GlmCV(CVSlectMixin, BaseEstimator):
    """
    Base class for generalized linear models tuned with cross-validation.

    Parameters
    ----------
    estimator: estimator object
        The base estimator to be cross-validated.

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

        Output
        ------
        self
            Fitted estimator.
        """

        # check the input data
        self._check_base_estimator()
        X, y, sample_weight =\
            self.estimator._validate_data(X=X, y=y,
                                          sample_weight=sample_weight)

        # Get the base estimator used for cross-validation and refitting.
        # This procedure may run prefitting routines such as fitting
        # an initial estimator to the data then update est.init
        est = self._get_estimator_for_cv(X=X, y=y, sample_weight=sample_weight)

        ###############################
        # Set tuning parameter values #
        ###############################

        # set the tuning values from the preprocessed data
        self._set_tuning_values(X=X, y=y, sample_weight=sample_weight,
                                estimator=est)
        # TODO the estimator arg here is used for adpt lasso, but
        # this is a bit subtle, perhaps streamline

        ########################
        # run cross-validation #
        ########################

        # maybe add sample weight to fit params
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
        else:
            fit_params = None

        # run cross-validation!
        # note subclasses should implement a _run_cv function
        cv_results, cv_runtime = self._run_cv(estimator=est, X=X, y=y,
                                              cv=self.cv,
                                              fit_params=fit_params)

        self.cv_results_ = cv_results
        self.cv_data_ = {'cv_runtime':  cv_runtime}

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)

        #########################################
        # refit with selected parameter setting #
        #########################################

        # set best tuning params
        est.set_params(**self.best_tune_params_)

        start_time = time()
        self.best_estimator_ = est.fit(X, y, sample_weight=sample_weight)
        self.cv_data_['refit_runtime'] = time() - start_time

        return self

    def _get_estimator_for_cv(self, X, y=None, sample_weight=None):
        """
        Output
        ------
        estimator
        """
        return clone(self.estimator)

    def predict(self, X):
        """
        See self.best_estimator_.predict
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """
        See self.best_estimator_.score
        """
        check_is_fitted(self)
        return self.best_estimator_.score(X, y)

    def decision_function(self, X):
        """
        See self.best_estimator_.decision_function
        """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    def predict_proba(self, X):
        """
        See self.best_estimator_.predict_proba
        """
        check_is_fitted(self)
        if not hasattr(self.best_estimator_, 'predict_proba'):
            raise NotImplementedError("This method does not have a preidct_proba function")
        else:
            return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """
        See self.best_estimator_.predict_log_proba
        """
        check_is_fitted(self)
        if not hasattr(self.best_estimator_, 'predict_log_proba'):
            raise NotImplementedError("This method does not have a predict_log_proba function")
        else:
            return self.best_estimator_.predict_log_proba(X)

    ################################
    # Subclasses need to implement #
    ################################

    def check_base_estimator(self):
        """
        Check the base estimator agrees with the CV class e.g. a Lasso object should be passed to LassoCV
        """
        # subclass should implement
        raise NotImplementedError

    def _set_tuning_values(self, X, y, sample_weight=None, estimator=None):
        """
        Sets the tuning parameter sequence for a given dataset.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        estimator: None
            An estimator to use for setting up the tuning values. If not provided, will use self.estimator. This is used for the adaptive lasso classes.

            TODO: this is a bit subtle -- perhaps streamline.
        """
        # subclass should overwrite
        raise NotImplementedError

    def get_tuning_sequence(self):
        """
        Returns the tuning parameters in an ordered sequence.

        Outout
        ------
        tuning_params: list
        """
        raise NotImplementedError

    def get_tuning_param_grid(self):
        """
        Returns tuning parameter grid.

        Output
        ------
        param_grid: dict of lists
        """
        raise NotImplementedError


class SinglePenSeqSetterMixin:
    """
    Sets the tuning parameter sequence for penalized GLMs with a single penalty parameter.

    Parameters
    ----------
    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence.

    pen_vals: None, array-like
        (Optional) User provided penalty value sequence. The penalty sequence should be monotonicly decreasing so the homotopy path algorithm works propertly.

    pen_min_mult: float
        Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.
    """

    def _set_tuning_values(self, X, y, sample_weight=None, estimator=None):
        """
        Sets the tuning parameter sequence for a given dataset. The tuning sequence is a either a linearly or logarithmically spaced sequence in the interval [pen_val_max * self.pen_min_mult, pen_val_max]. Logarithmic spacing means the penalty values are more dense close to zero.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        estimator: None
            An estimator to use for setting up the tuning values. If not provided, will use self.estimator. This is used for the adaptive lasso classes.

            TODO: this is a bit subtle -- perhaps streamline.
        """
        # TODO: this is a little subtle, perhaps streamline this
        if estimator is None:
            estimator = self.estimator

        if self.pen_vals is None:
            pen_val_max = estimator.\
                get_pen_val_max(X=X, y=y, sample_weight=sample_weight)

            pen_val_seq = \
                get_sequence_decr_max(max_val=pen_val_max,
                                      min_val_mult=self.pen_min_mult,
                                      num=self.n_pen_vals,
                                      spacing=self.pen_spacing)

        else:
            pen_val_seq = np.array(self.pen_vals)

        # ensure decreasing
        self.pen_val_seq_ = np.sort(pen_val_seq)[::-1]

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values.

        Output
        ------
        param_seq: list of dicts
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
