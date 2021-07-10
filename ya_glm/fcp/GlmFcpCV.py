from time import time
from copy import deepcopy
from sklearn.base import clone

from ya_glm.processing import check_estimator_type

from ya_glm.GlmCV import GlmCVSinglePen, get_pen_val_seq
from ya_glm.cv.CVGridSearch import CVGridSearchMixin
from ya_glm.fcp.GlmFcp import GlmFcp


class GlmFcpCV(CVGridSearchMixin, GlmCVSinglePen):

    def fit(self, X, y):

        # check the input data
        self._check_base_estimator(self.estimator)
        est = clone(self.estimator)
        X, y = est._validate_data(X, y)

        # get initialization from raw data
        init_data = est.get_init_data(X, y)
        if 'est' in init_data:
            self.init_est_ = init_data['est']
            del init_data['est']

        # make sure all estimators we tune over have the same initialization
        # TODO: do we really need to copy here?
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

        # set best tuning params
        est.set_params(**self.best_tune_params_)

        # refit on the raw data
        start_time = time()
        est.fit(X, y)
        self.cv_data_['refit_runtime'] = time() - start_time
        self.best_estimator_ = est

        return self

    def _set_tuning_values(self, X, y, init_data):

        if self.pen_vals is None:
            pen_val_max = self.estimator.get_pen_val_max(X, y, init_data)
        else:
            pen_val_max = None

        self.pen_val_seq_ = \
            get_pen_val_seq(pen_val_max,
                            n_pen_vals=self.n_pen_vals,
                            pen_vals=self.pen_vals,
                            pen_min_mult=self.pen_min_mult,
                            pen_spacing=self.pen_spacing)

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmFcp)
