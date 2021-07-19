from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid
import numpy as np
from time import time
from numbers import Number
from textwrap import dedent

from ya_glm.autoassign import autoassign
from ya_glm.init_signature import add_from_classes
from ya_glm.make_docs import merge_param_docs
from ya_glm.pen_seq import get_pen_val_seq, get_enet_pen_val_seq, \
    get_enet_ratio_seq
from ya_glm.cv.cv_select import CVSlectMixin  # select_best_cv_tune_param


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

        # run cross-validation on the raw data
        start_time = time()
        self.cv_results_ = \
            self._run_cv(estimator=est, X=X, y=y, cv=self.cv,
                         fit_params=fit_params)

        self.cv_data_ = {'cv_runtime':  time() - start_time}

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)

        # set best tuning params
        est.set_params(**self.best_tune_params_)

        # refit on the raw data
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


_enet_cv_params = dedent("""
l1_ratio: float, str, list
    The l1_ratio value to use. If a float is provided then this parameter is fixed and not tuned over. If l1_ratio='tune' then the l1_ratio is tuned over using an automatically generated tuning parameter sequence. Alternatively, the user may provide a list of l1_ratio values to tune over.

n_l1_ratio_vals: int
    Number of l1_ratio values to tune over. The l1_ratio tuning sequence is a logarithmically spaced grid of values between 0 and 1 that has more values close to 1.

l1_ratio_min:
    The smallest l1_ratio value to tune over.
""")


class GlmCVENet(GlmCVSinglePen):
    """
    Base class for Elastic Net penalized GLMs tuned with cross-validation.
    """
    _param_descr = merge_param_docs(GlmCVSinglePen._params_descr,
                                    _pen_seq_params)

    @add_from_classes(GlmCVSinglePen, add_first=False)
    def __init__(self,
                 # pen_min_mult=1e-4,  # make this more extreme for enet
                 l1_ratio=0.5,
                 n_l1_ratio_vals=10,
                 l1_ratio_min=0.1,
                 ):
        pass

    def _tune_l1_ratio(self):
        """
        Output
        ------
        yes_tune_l1_ratio: bool
            Whether or not we tune the l1_ratio parameter.
        """
        # Do we tune the l1_ratio
        if self.l1_ratio == 'tune' or hasattr(self.l1_ratio, '__len__'):
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

    def _set_tuning_values(self, X, y, sample_weight=None):
        if self.pen_vals is None:
            enet_pen_max = self.estimator.\
                get_pen_val_max(X, y, sample_weight=sample_weight)
            lasso_pen_max = enet_pen_max * self.estimator.l1_ratio
        else:
            lasso_pen_max = None

        self._set_tune_from_lasso_max(X, y, lasso_pen_max=lasso_pen_max)

    def _set_tune_from_lasso_max(self, X, y, lasso_pen_max=None):
        """
        Sets the ElasticNet tuning sequence given the largest reasonable lasso penalty value.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        lasso_pen_max: float
            The lasso penalty max value
        """

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
