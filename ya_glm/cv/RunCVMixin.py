from time import time

from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from functools import partial

from ya_glm.cv.run_cv import run_cv_grid, run_cv_path, \
    add_params_to_cv_results, score_from_fit_path


class RunCVGridMixin:

    def _run_cv(self, estimator, X, y=None, cv=None, fit_params=None):
        """
        Runs cross-validation using _run_cv_grid
        """

        start_time = time()
        cv_results = self._run_cv_grid(estimator=estimator, X=X, y=y, cv=cv,
                                       fit_params=fit_params)
        cv_runtime = time() - start_time

        return cv_results, cv_runtime

    def _run_cv_grid(self, estimator, X, y=None,
                     cv=None, fit_params=None):
        """
        Runs cross-validation using sklearn's GridSearchCV.

        Parameters
        ----------
        estimator:
            The base estimator to used for cross-validation.

        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        cv: int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy. See sklearn.model_selection.GridSearchCV

        fit_params: dict
            Additional fit parameters e.g. sample_weight.

        Output
        ------
        cv_results: dict
            The cross-validation results. See GridSearchCV.cv_results_
        """
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


class RunCVGridOrPathMixin:

    def _run_cv(self, estimator, X, y=None, cv=None, fit_params=None):
        start_time = time()

        if estimator._get_solver().has_path_algo:
            # if we have a path algorithm available use it

            cv_results = \
                self._run_cv_path(estimator=estimator, X=X, y=y, cv=cv,
                                  fit_params=fit_params)

        else:
            # otherwise just do grid search
            cv_results = \
                self._run_cv_grid(estimator=estimator, X=X, y=y, cv=cv,
                                  fit_params=fit_params)

        cv_runtime = time() - start_time

        return cv_results, cv_runtime

    def _run_cv_path(self, estimator, X, y=None, cv=None, fit_params=None):
        """
        Runs cross-validation using a path algorithm.

        Parameters
        ----------
        estimator:
            The base estimator to used for cross-validation.

        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        cv: int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy. See sklearn.model_selection.GridSearchCV

        fit_params: dict
            Additional fit parameters e.g. sample_weight.

        Output
        ------
        cv_results: dict
            The cross-validation results. See GridSearchCV.cv_results_
        """

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        pen_seq_config = self._get_penalty_seq_config(estimator)

        solve_path_kws = {'loss': estimator._get_loss_config(),
                          'penalty_seq': pen_seq_config,
                          'fit_intercept': estimator.fit_intercept
                          }

        cv_results, _ = \
            run_cv_path(X=X, y=y,
                        fold_iter=cv.split(X, y),
                        fit_and_score_path=fit_and_score_path,
                        solve_path_kws=solve_path_kws,
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
        est = clone(estimator)  # TODO: do we want to clone here?
        def est_from_fit(fit_out, pre_pro_out):
            est._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
            return est

        # get solve path from the estimator
        solve_path = est._get_solver().solve_path

        fit_and_score_path = partial(score_from_fit_path,
                                     solve_path=solve_path,
                                     est_from_fit=est_from_fit,
                                     scorer=self.cv_scorer,
                                     preprocess=est.preprocess)

        return fit_and_score_path
