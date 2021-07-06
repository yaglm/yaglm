from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid
import numpy as np
from time import time
from numbers import Number
from textwrap import dedent

from ya_glm.autoassign import autoassign
from ya_glm.utils import get_sequence_decr_max, get_enet_ratio_seq


from ya_glm.cv.cv_select import CVSlectMixin  # select_best_cv_tune_param


# _cv_params = dedent(
# """
# cv: int, cross-validation generator or an iterable, default=None
#     Determines the cross-validation splitting strategy.

# cv_select_metric: None, str
#     Which metric to use for select the best tuning parameter if multiple metrics are computed.

# cv_scorer: None, callable(est, X, y) -> dict or float
#     A function for evaluating the cross-validation fit estimators.
#     If this returns a dict of multiple scores then cv_select_metric determines which metric is used to select the tuning parameter.

# cv_n_jobs: None, int
#     Number of jobs to run in parallel.

# cv_verbose: int
#     Amount of printout during cross-validation.

# cv_pre_dispatch: int, or str, default=n_jobs
#     Controls the number of jobs that get dispatched during parallel execution
# """
# )


class GlmCV(CVSlectMixin, BaseEstimator):

    @autoassign
    def __init__(self,
                 fit_intercept=True, standardize=False, opt_kws={},

                 cv=None,
                 cv_select_rule='best',
                 cv_select_metric=None,
                 cv_scorer=None,
                 cv_verbose=0, cv_n_jobs=None,
                 cv_pre_dispatch='2*n_jobs'):
        pass

    def fit(self, X, y):
        """
        Runs cross-validation then refits the GLM with the selected tuning parameter.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.
        """

        # check the input data
        est = self._get_base_estimator()
        X, y = est._validate_data(X, y)

        # set up the tuning parameter values using the processed data
        self._set_tuning_values(X=X, y=y)

        # run cross-validation on the raw data
        start_time = time()
        self.cv_results_ = self._run_cv(X=X, y=y, cv=self.cv)
        self.cv_data_ = {'cv_runtime':  time() - start_time}

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)
        est.set_params(**self.best_tune_params_)

        # refit on the raw data
        start_time = time()
        est.fit(X, y)
        self.cv_data_['refit_runtime'] = time() - start_time
        self.best_estimator_ = est

        return self

    def _set_tuning_values(self, X, y):
        """
        Sets the tuning parameter sequence from the transformed data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The processed training covariate data.

        y: array-like, shape (n_samples, )
            The processed training response data.
        """
        # subclass should overwrite
        raise NotImplementedError

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values.
        Make sure the method that computes the cross-validation results
        orders the parameters in the same order as get_tuning_sequence()

        Output
        ------
        values: iterable
        """
        param_grid = self.get_tuning_param_grid()
        return list(ParameterGrid(param_grid))

    def get_tuning_param_grid(self):
        """
        Returns parameter grid.

        Output
        ------
        dol:
        """
        # subclass should overwrite
        raise NotImplementedError

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

    def _get_base_estimator(self):
        """
        Output
        ------
        est:
            The base estimator with all parameters set except those
            that should be selected with cross-validation.
        """
        c = self._get_base_class()
        p = self._get_base_est_params()
        return c(**p)

    def _get_base_class(self):
        raise NotImplementedError

    def _get_base_est_params(self):
        raise NotImplementedError

    # properties inhereted from base_estimator
    @property
    def _model_type(self):
        return self._get_base_estimator()._model_type

# GlmCV.__doc__ = dedent(
#     """
#     Base class for generalized linear models tuned with cross-validation.

#     Parameters
#     ----------
#     {}
#     """.format(_cv_params)
# )


# _cv_pen_params = dedent("""

# n_pen_vals: int
#     Number of penalty values to try for automatically generated tuning parameter sequence.

# pen_vals: None, array-like
#     (Optional) User provided penalty value sequence. The penalty sequence should be monotonicly decreasing so the homotopy path algorithm works propertly.

# pen_min_mult: float
#     Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

# pen_spacing: str
#     How the penalty values are spaced. Must be one of ['log', 'lin']
#     for logarithmic and linear spacing respectively.
# """)


class GlmCVSinglePen(GlmCV):

    @autoassign
    def __init__(self,
                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log',
                 **cv_kws):
        super().__init__(**cv_kws)
        pass

    def _set_tuning_values(self, X, y):

        if self.pen_vals is None:

            # get largest penalty value
            est = self._get_base_estimator()
            pen_val_max = est.get_pen_val_max(X, y)

            pen_val_seq = get_sequence_decr_max(max_val=pen_val_max,
                                                min_val_mult=self.pen_min_mult,
                                                num=self.n_pen_vals,
                                                spacing=self.pen_spacing)
        else:
            pen_val_seq = np.array(self.pen_vals)

        self.pen_val_seq_ = np.sort(pen_val_seq)[::-1]  # ensure decreasing

    def get_tuning_param_grid(self):
        return {'pen_val': self.pen_val_seq_}


# GlmCVSinglePen.__doc__ = dedent(
#     """
#     Base class for penalized generalized linear model tuned with cross-validation.

#     Parameters
#     ----------
#     {}

#     fit_intercept: bool
#         Whether or not to fit an intercept.

#     weights: None, array-like shape (n_features, )
#         (Optinal) Feature weights for penalty.

#     opt_kws: dict
#         Key word arguments to the optimization algorithm.

#     {}
#     """.format(_cv_pen_params, _cv_params)
# )


# _enet_cv_params = dedent("""

# l1_ratio: float, str, list
#     The l1_ratio value to use. If a float is provided then this parameter is fixed and not tuned over. If l1_ratio='tune' then the l1_ratio is tuned over using an automatically generated tuning parameter sequence. Alternatively, the user may provide a list of l1_ratio values to tune over.

# n_l1_ratio_vals: int
#     Number of l1_ratio values to tune over. The l1_ratio tuning sequence is a logarithmically spaced grid of values between 0 and 1 that has more values close to 1.

# l1_ratio_min:
#     The smallest l1_ratio value to tune over.
# """)


class GlmCVENet(GlmCV):

    @autoassign
    def __init__(self,
                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-4,  # make this more extreme for enet
                 pen_spacing='log',

                 l1_ratio=0.5,
                 n_l1_ratio_vals=10,
                 l1_ratio_min=0.1,

                 **cv_kws
                 ):
        super().__init__(**cv_kws)
        pass

    def _get_base_est_params(self):

        # params = {'fit_intercept': self.fit_intercept,
        #           'opt_kws': self.opt_kws,
        #           **self._get_extra_base_params()}

        params = self._get_extra_base_params()

        if not self._tune_l1_ratio():
            params['l1_ratio'] = self.l1_ratio

        if not self._tune_pen_val():
            params['pen_val'] = self.pen_vals

        return params

    def _get_extra_base_params(self):
        """

        """
        raise NotImplementedError

    def _tune_l1_ratio(self):
        """
        Output
        ------
        yes_tune_l1_ratio: bool
            Whether or not we tune the l1_ratio parameter.
        """
        # Do we tune the l1_ratio
        if self.l1_ratio is None or \
                self.l1_ratio == 'tune' or\
                not isinstance(self.l1_ratio, Number):

            return True
        else:
            return False

    def _tune_pen_val(self):
        """
        Output
        ------
        yes_tune_pen_val: bool
            Whether or not we tune the pen_val parameter.
        """

        # Do we tune the pen_vals
        if isinstance(self.pen_vals, Number):
            return False
        else:
            return True

    def _set_tuning_values(self, X, y):

        ##################################
        # setup l1_ratio tuning sequence #
        ##################################
        if self._tune_l1_ratio():

            if self.l1_ratio is not None and not self.l1_ratio == 'tune':
                # user specified values
                l1_ratio_seq = np.array(self.l1_ratio).reshape(-1)

            else:
                # otherwise set these values by default
                l1_ratio_seq = \
                    get_enet_ratio_seq(min_val=self.l1_ratio_min,
                                       num=self.n_l1_ratio_vals)

        else:
            l1_ratio_val = self.l1_ratio

        #################################
        # setup pen_val tuning sequence #
        #################################
        if self._tune_pen_val():

            if self.pen_vals is not None:  # user provided pen vals
                pen_vals = np.array(self.pen_vals)

            else:  # automatically derive tuning sequence

                # if the l1_ratio is ever zero our default
                # pen_val_seq will fail
                if self._tune_l1_ratio():
                    l1_ratio_min = min(l1_ratio_seq)
                else:
                    l1_ratio_min = l1_ratio_val
                if l1_ratio_min <= np.finfo(float).eps:
                    raise ValueError("Unable to set pen_val_seq using default"
                                     "when the l1_ratio is zero."
                                     " Either change thel1_ratio, or "
                                     "input a sequence of pen_vals yourself!")

                # get largest Lasso penalty value for this loss function
                # make sure we get set tuning parameters using the processed data
                # the optimization algorithm will actually see
                est = self._get_base_estimator()
                est.set_params(l1_ratio=1)
                lasso_pen_val_max = est.get_pen_val_max(X, y)

                if self._tune_l1_ratio():
                    # setup grid of pen vals for each l1 ratio

                    n_l1_ratio_vals = len(l1_ratio_seq)
                    pen_vals = np.zeros((n_l1_ratio_vals, self.n_pen_vals))
                    for l1_idx in range(n_l1_ratio_vals):

                        # largest pen val for ElasticNet given this l1_ratio
                        max_val = lasso_pen_val_max / l1_ratio_seq[l1_idx]

                        pen_vals[l1_idx, :] = \
                            get_sequence_decr_max(max_val=max_val,
                                                  min_val_mult=self.pen_min_mult,
                                                  num=self.n_pen_vals,
                                                  spacing=self.pen_spacing)

                else:
                    # setup pen val sequence

                    max_val = lasso_pen_val_max / l1_ratio_val
                    pen_vals = \
                        get_sequence_decr_max(max_val=max_val,
                                              min_val_mult=self.pen_min_mult,
                                              num=self.n_pen_vals,
                                              spacing=self.pen_spacing)

        # store data
        if self._tune_l1_ratio() and self._tune_pen_val():
            assert pen_vals.ndim == 2
            assert pen_vals.shape[0] == len(l1_ratio_seq)

            # make sure pen vals are always in decreasing order
            for l1_idx in range(pen_vals.shape[0]):
                pen_vals[l1_idx, :] = np.sort(pen_vals[l1_idx, :])[::-1]

            self.pen_val_seq_ = pen_vals
            self.l1_ratio_seq_ = l1_ratio_seq

        elif self._tune_pen_val():
            # make sure pen vals are always in decreasing order
            pen_vals = np.sort(np.array(pen_vals))[::-1]
            self.pen_val_seq_ = pen_vals.reshape(-1)

        elif self._tune_l1_ratio():
            self.l1_ratio_seq_ = l1_ratio_seq

    def get_tuning_param_grid(self):
        if self._tune_l1_ratio() and self._tune_pen_val():
            return self.get_tuning_sequence()

        elif self._tune_l1_ratio():
            return {'l1_ratio': self.l1_ratio_seq_}

        elif self._tune_pen_val():
            return {'pen_val': self.pen_val_seq_}

    def get_tuning_sequence(self):
        """
        Returns a list of tuning parameter values.

        Output
        ------
        values: iterable
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


# GlmCVENet.__doc__ = dedent(
#     """
#     Elastic Net penalized generalized linear model tuned with cross-validation.

#     Parameters
#     ----------
#     {}

#     {}

#     fit_intercept: bool
#         Whether or not to fit an intercept.

#     tikhonov: None, array-like shape (n_features, n_features)
#         (Optinal) Tikhonov matrix.

#     opt_kws: dict
#         Key word arguments to the optimization algorithm.

#     {}
#     """.format(_cv_pen_params, _enet_cv_params, _cv_params)
# )
