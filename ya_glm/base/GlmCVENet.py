from sklearn.base import is_classifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import check_cv

import numpy as np
from numbers import Number
from textwrap import dedent

from ya_glm.base.GlmCV import GlmCV, _pen_seq_params
from ya_glm.init_signature import add_from_classes
from ya_glm.make_docs import merge_param_docs
from ya_glm.pen_seq import get_enet_pen_val_seq, \
    get_enet_ratio_seq
from ya_glm.cv.run_cv import run_cv_path, \
    add_params_to_cv_results


_enet_cv_params = _pen_seq_params + dedent("""
l1_ratio: float, str, list
    The l1_ratio value to use. If a float is provided then this parameter is fixed and not tuned over. If l1_ratio='tune' then the l1_ratio is tuned over using an automatically generated tuning parameter sequence. Alternatively, the user may provide a list of l1_ratio values to tune over.

n_l1_ratio_vals: int
    Number of l1_ratio values to tune over. The l1_ratio tuning sequence is a logarithmically spaced grid of values between 0 and 1 that has more values close to 1.

l1_ratio_min:
    The smallest l1_ratio value to tune over.
""")


class GlmCVENet(GlmCV):
    """
    Base class for Elastic Net penalized GLMs tuned with cross-validation.
    """
    _param_descr = merge_param_docs(GlmCV._params_descr)

    @add_from_classes(GlmCV, add_first=False)
    def __init__(self,
                 n_pen_vals=100,
                 pen_vals=None,
                 pen_min_mult=1e-3,
                 pen_spacing='log',
                 l1_ratio=0.5,
                 n_l1_ratio_vals=10,
                 l1_ratio_min=0.1,
                 ): pass

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

    def _run_cv_path(self, estimator, X, y=None, cv=None, fit_params=None):
        # TODO: see if we can simplify this with self.get_tuning_sequence()

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        kws = self._get_solve_path_enet_base_kws()

        if self._tune_l1_ratio() and self._tune_pen_val():
            # Tune over both l1_ratio and pen_val
            # for each l1_ratio, fit the pen_val path

            all_cv_results = []
            for l1_idx, l1_ratio in enumerate(self.l1_ratio_seq_):

                # set pen vals for this sequence
                pen_val_seq = self.pen_val_seq_[l1_idx, :]
                kws['lasso_pen_seq'] = pen_val_seq * l1_ratio
                kws['ridge_pen_seq'] = pen_val_seq * (1 - l1_ratio)

                # fit path
                cv_res, _ = \
                    run_cv_path(X=X, y=y,
                                fold_iter=cv.split(X, y),
                                fit_and_score_path=fit_and_score_path,
                                kws=kws,
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

            kws['lasso_pen_seq'] = pen_val_seq * l1_ratio_seq
            kws['ridge_pen_seq'] = pen_val_seq * (1 - l1_ratio_seq)

            # fit path
            cv_results, _ = \
                run_cv_path(X=X, y=y,
                            fold_iter=cv.split(X, y),
                            fit_and_score_path=fit_and_score_path,
                            kws=kws,
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

    def _get_solve_path_enet_base_kws(self):
        """
        Returns the key word argus for solve_path excluding the lasso_pen_seq and L2_pen_seq values.
        """
        raise NotImplementedError
