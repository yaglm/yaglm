import numpy as np
from sklearn.model_selection import GridSearchCV

from ya_glm.cv.cv_select import add_se


class CVGridSearchMixin:

    def _run_cv(self, estimator, X, y=None, cv=None, fit_params=None):

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


def run_cv_grid(X, y, estimator, param_grid, cv,
                scoring=None, fit_params=None,
                n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    """
    Runs cross-validation using sklearn.model_selection.GridSearchCV.
    Mostly see documentation of GridSearchCV

    Parameters
    ----------
    TODO

    Output
    ------
    cv_results: dict
    """
    cv_est = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          scoring=scoring,
                          n_jobs=n_jobs,
                          refit=False,
                          cv=cv,
                          verbose=verbose,
                          pre_dispatch=pre_dispatch,
                          error_score=np.nan,
                          return_train_score=True)

    fit_params = fit_params if fit_params is not None else {}
    cv_est.fit(X, y, **fit_params)

    cv_results = add_se(cv_est.cv_results_)
    # TODO: maybe remove splits
    # TODO: maybe remove std
    return cv_results
