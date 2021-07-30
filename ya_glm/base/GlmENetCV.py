from sklearn.base import is_classifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import check_cv

import numpy as np
from numbers import Number

from ya_glm.PenaltyConfig import ConvexPenaltySequence

from ya_glm.cv.RunCVMixin import RunCVGridOrPathMixin
from ya_glm.cv.run_cv import run_cv_path, add_params_to_cv_results

from ya_glm.pen_seq import get_enet_pen_val_seq, \
    get_enet_ratio_seq


class GlmENetCVMixin(RunCVGridOrPathMixin):
    """
    Mixin for Elastic Net penalized GLMs tuned with cross-validation.
    This mixin takes care of both running cross-validation with path algorithms and setting up the tuning parameter sequence.


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

    l1_ratio: float, str, list
        The l1_ratio value to use. If a float is provided then this parameter is fixed and not tuned over. If l1_ratio='tune' then the l1_ratio is tuned over using an automatically generated tuning parameter sequence. Alternatively, the user may provide a list of l1_ratio values to tune over.

    n_l1_ratio_vals: int
        Number of l1_ratio values to tune over. The l1_ratio tuning sequence is a logarithmically spaced grid of values between 0 and 1 that has more values close to 1.

    l1_ratio_min:
        The smallest l1_ratio value to tune over.
    """

    def _tune_l1_ratio(self):
        """
        Whether or not we tune the l1_ratio parameter.

        Output
        ------
        yes_tune_l1_ratio: bool
        """
        # Do we tune the l1_ratio
        if self.l1_ratio == 'tune' or hasattr(self.l1_ratio, '__len__'):
            return True
        else:
            return False

    def _tune_pen_val(self):
        """
        Whether or not we tune the pen_val parameter
        Output
        ------
        yes_tune_pen_val: bool
        """

        # Do we tune the pen_vals
        if isinstance(self.pen_vals, Number):
            return False
        else:
            return True

    def _set_tuning_values(self, X, y, sample_weight=None, estimator=None):
        """

        Sets the tuning parameter sequence for a given dataset.

        The pen_vals are either linearly or logarithmically spaced. The l1_ratio values (if tuned) are always linearly spaced.s

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
            enet_pen_max = estimator.\
                get_pen_val_max(X=X, y=y, sample_weight=sample_weight)
            lasso_pen_max = enet_pen_max * estimator.l1_ratio
        else:
            lasso_pen_max = None

        ##################################
        # setup l1_ratio tuning sequence #
        ##################################
        if self._tune_l1_ratio():
            l1_ratio_val = None

            if self.l1_ratio is not None and not self.l1_ratio == 'tune':
                # user specified values
                l1_ratio_seq = np.array(self.l1_ratio).reshape(-1)

            else:
                # otherwise set these values by default
                l1_ratio_seq = \
                    get_enet_ratio_seq(min_val=self.l1_ratio_min,
                                       num=self.n_l1_ratio_vals)

            self.l1_ratio_seq_ = l1_ratio_seq

        else:
            l1_ratio_val = self.l1_ratio
            l1_ratio_seq = None

        #################################
        # setup pen_val tuning sequence #
        #################################

        if self._tune_pen_val():

            self.pen_val_seq_ = \
                get_enet_pen_val_seq(lasso_pen_val_max=lasso_pen_max,
                                     pen_vals=self.pen_vals,
                                     n_pen_vals=self.n_pen_vals,
                                     pen_min_mult=self.pen_min_mult,
                                     pen_spacing=self.pen_spacing,
                                     l1_ratio_seq=l1_ratio_seq,
                                     l1_ratio_val=l1_ratio_val)

    def get_tuning_param_grid(self):
        """
        Returns tuning parameter grid.

        Output
        ------
        param_grid: dict of lists
        """
        if self._tune_l1_ratio() and self._tune_pen_val():
            return self.get_tuning_sequence()

        elif self._tune_l1_ratio():
            return {'l1_ratio': self.l1_ratio_seq_}

        elif self._tune_pen_val():
            return {'pen_val': self.pen_val_seq_}

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values. When both l1_ratio and pen_val are tuned over we iterate first over l1_ratio values (outter loop) then over pen_vals (inner loop).

        Output
        ------
        values: list of dicts
        """
        if self._tune_l1_ratio() and self._tune_pen_val():
            n_l1_ratios, n_pen_vals = self.pen_val_seq_.shape

            # outer loop over l1_ratios, inner loop over pen_vals
            param_list = []
            for l1_idx in range(n_l1_ratios):
                l1_ratio_val = self.l1_ratio_seq_[l1_idx]

                for pen_idx in range(n_pen_vals):
                    pen_val = self.pen_val_seq_[l1_idx, pen_idx]

                    param_list.append({'l1_ratio': l1_ratio_val,
                                       'pen_val': pen_val})

            return param_list

        elif self._tune_l1_ratio():
            param_grid = {'l1_ratio': self.l1_ratio_seq_}
            return list(ParameterGrid(param_grid))

        elif self._tune_pen_val():
            param_grid = {'pen_val': self.pen_val_seq_}
            return list(ParameterGrid(param_grid))

    ###########################
    # CV path for elastic net #
    ###########################
    def _run_cv_path(self, estimator, X, y=None, cv=None, fit_params=None):
        """
        Runs cross-validation for the ElasticNet penalty when we have an available path algorithm.

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
        # TODO: see if we can simplify this with self.get_tuning_sequence()

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        # setup base penalty config
        # i.e. everything but lasso_pen and ridge_pen
        base_penalty = estimator._get_penalty_config()
        base_penalty.lasso_pen_val = None
        base_penalty.ridge_pen_val = None

        solve_path_kws = {'loss': estimator._get_loss_config(),
                          'fit_intercept': estimator.fit_intercept}

        if self._tune_l1_ratio() and self._tune_pen_val():
            # Tune over both l1_ratio and pen_val
            # for each l1_ratio, fit the pen_val path

            all_cv_results = []
            for l1_idx, l1_ratio in enumerate(self.l1_ratio_seq_):

                # set pen vals for this sequence
                pen_val_seq = self.pen_val_seq_[l1_idx, :]
                lasso_pen_seq = pen_val_seq * l1_ratio
                ridge_pen_seq = pen_val_seq * (1 - l1_ratio)

                pen_seq_config = \
                    ConvexPenaltySequence(penalty=base_penalty,
                                          lasso_pen_seq=lasso_pen_seq,
                                          ridge_pen_seq=ridge_pen_seq)

                solve_path_kws['penalty_seq'] = pen_seq_config

                # fit path
                cv_res, _ = \
                    run_cv_path(X=X, y=y,
                                fold_iter=cv.split(X, y),
                                fit_and_score_path=fit_and_score_path,
                                solve_path_kws=solve_path_kws,
                                fit_params=fit_params,
                                include_spilt_vals=False,  # maybe make this True?
                                add_params=False,
                                n_jobs=self.cv_n_jobs,
                                verbose=self.cv_verbose,
                                pre_dispatch=self.cv_pre_dispatch)

                all_cv_results.append(cv_res)

            # combine cv_results for each l1_ratio value
            cv_results = {}
            n_l1_ratios = len(self.l1_ratio_seq_)
            for name in all_cv_results[0].keys():
                cv_results[name] = \
                    np.concatenate([all_cv_results[i][name]
                                    for i in range(n_l1_ratios)])

        else:
            # only tune over one of l1_ratio or pen_val

            # setup L1/L2 penalty sequence
            if self._tune_pen_val():
                # tune over pen_val
                pen_val_seq = self.pen_val_seq_
                l1_ratio_seq = self.l1_ratio  # * np.ones_like(pen_val_seq)

            elif self._tune_l1_ratio():
                # tune over l1_ratio
                l1_ratio_seq = self.l1_ratio_seq_
                pen_val_seq = self.pen_val  # * np.ones_like(pen_val_seq)

            lasso_pen_seq = pen_val_seq * l1_ratio_seq
            ridge_pen_seq = pen_val_seq * (1 - l1_ratio_seq)

            pen_seq_config = \
                ConvexPenaltySequence(penalty=base_penalty,
                                      lasso_pen_seq=lasso_pen_seq,
                                      ridge_pen_seq=ridge_pen_seq)

            solve_path_kws['penalty_seq'] = pen_seq_config

            # fit path
            cv_results, _ = \
                run_cv_path(X=X, y=y,
                            fold_iter=cv.split(X, y),
                            fit_and_score_path=fit_and_score_path,
                            solve_path_kws=solve_path_kws,
                            fit_params=fit_params,
                            include_spilt_vals=False,  # maybe make this True?
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
