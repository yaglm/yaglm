from time import time
import numpy as np
from copy import deepcopy

from ya_glm.utils import get_sequence_decr_max
from ya_glm.autoassign import autoassign

from ya_glm.GlmCV import GlmCVSinglePen
from ya_glm.cv.CVGridSearch import CVGridSearchMixin


class GlmFcpCV(CVGridSearchMixin, GlmCVSinglePen):

    @autoassign
    def __init__(self, pen_func='scad', pen_func_kws={}, init='default',
                 **kws):
        super().__init__(**kws)

    def fit(self, X, y):
        # validate the data!
        est = self._get_base_estimator()
        X, y = est._validate_data(X, y)

        # get initialization from raw data
        init_data = est.get_init_data(X, y)
        if 'est' in init_data:
            self.init_est_ = init_data['est']

        est.set_params(init=deepcopy(init_data))

        # set up the tuning parameter values
        self._set_tuning_values(X=X, y=y, init_data=init_data)

        # run cross-validation
        self.cv_data_ = {}
        start_time = time()
        self.cv_results_ = self._run_cv(X=X, y=y, cv=self.cv, estimator=est)
        self.cv_data_['cv_runtime'] = time() - start_time

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

    def _set_tuning_values(self, X, y, init_data):

        if self.pen_vals is None:

            est = self._get_base_estimator()
            pen_val_max = est.get_pen_val_max(X, y, init_data)

            pen_vals = get_sequence_decr_max(max_val=pen_val_max,
                                             min_val_mult=self.pen_min_mult,
                                             num=self.n_pen_vals,
                                             spacing=self.pen_spacing)
        else:
            pen_vals = np.array(self.pen_vals)

        self.pen_val_seq_ = np.sort(pen_vals)[::-1]  # ensure decreasing


class GlmFcpFitLLACV(GlmFcpCV):

    @autoassign
    def __init__(self, lla_n_steps=1, lla_kws={},
                 **kws):
        GlmFcpCV.__init__(self, **kws)

    def _get_base_est_params(self):
        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'pen_func': self.pen_func,
                'pen_func_kws': self.pen_func_kws,
                'init': self.init,

                'lla_n_steps': self.lla_n_steps,
                'lla_kws': self.lla_kws
                }
