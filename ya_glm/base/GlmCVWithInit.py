from time import time
from sklearn.base import clone

from ya_glm.base.GlmCV import GlmCVSinglePen, GlmCVENet


class CVWithInitMixin:

    def fit(self, X, y, sample_weight=None):

        # check the input data
        self._check_base_estimator(self.estimator)
        est = clone(self.estimator)
        X, y, sample_weight = est._validate_data(X, y,
                                                 sample_weight=sample_weight)

        # get initialization from raw data
        init_data = est.get_init_data(X, y, sample_weight=sample_weight)
        if 'est' in init_data:
            self.init_est_ = init_data['est']
            del init_data['est']

        # make sure all estimators we tune over have the same initialization
        est.set_params(init=init_data)

        # possibly do something else before computing the tuning path
        # TODO: we do this for adaptive lasso, but this is a bit ugly
        # can we come up with a better soltuion for setting adaptive
        # weights for adpative lasso?
        est = self._pre_fit(X=X, y=y, init_data=init_data, estimator=est,
                            sample_weight=sample_weight)

        # set up the tuning parameter values
        self._set_tuning_values(X=X, y=y,
                                init_data=init_data,
                                estimator=est,
                                sample_weight=sample_weight)

        # maybe add sample weight to fit params
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
        else:
            fit_params = None

        # run cross-validation
        self.cv_data_ = {}
        start_time = time()
        self.cv_results_ = self._run_cv(estimator=est, X=X, y=y, cv=self.cv,
                                        fit_params=fit_params)
        self.cv_data_['cv_runtime'] = time() - start_time

        # select best tuning parameter values
        self.best_tune_idx_, self.best_tune_params_ = \
            self._select_tune_param(self.cv_results_)

        # set best tuning params
        est.set_params(**self.best_tune_params_)

        # refit on the raw data
        start_time = time()
        est.fit(X, y, sample_weight=sample_weight)
        self.cv_data_['refit_runtime'] = time() - start_time
        self.best_estimator_ = est

        return self

    def _pre_fit(self, X, y, init_data, estimator, sample_weight=None):
        """
        This does nothing, but a subclass may overwrite
        """
        return estimator

    def _set_tuning_values(self, X, y, init_data, estimator):
        raise NotImplementedError


class GlmCVWithInitSinglePen(CVWithInitMixin, GlmCVSinglePen):

    def _set_tuning_values(self, X, y, init_data, estimator,
                           sample_weight=None):
        if self.pen_vals is None:
            pen_val_max = \
                estimator.get_pen_val_max(X, y, init_data,
                                          sample_weight=sample_weight)
        else:
            pen_val_max = None

        self._set_tune_from_pen_max(pen_val_max=pen_val_max)


class GlmCVWithInitENet(CVWithInitMixin, GlmCVENet):

    def _set_tuning_values(self, X, y, init_data, estimator,
                           sample_weight=None):

        if self.pen_vals is None:
            enet_pen_max = \
                estimator.get_pen_val_max(X, y, init_data,
                                          sample_weight=sample_weight)
            lasso_pen_max = enet_pen_max * estimator.l1_ratio
        else:
            lasso_pen_max = None

        self._set_tune_from_lasso_max(X, y, lasso_pen_max=lasso_pen_max)
