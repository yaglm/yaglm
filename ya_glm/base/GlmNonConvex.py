import numpy as np
from sklearn.base import clone

from ya_glm.base.Glm import Glm
from ya_glm.base.InitFitMixin import InitFitMixin
from ya_glm.pen_max.non_convex import get_pen_max
from ya_glm.processing import process_init_data
from ya_glm.base.GlmCV import GlmCV


class GlmNonConvex(InitFitMixin, Glm):
    """
    Base class for GLMs with non-convex penalties.
    """

    def prefit(self, X, y, sample_weight=None):
        """
        Fits an initial estimator to the data, computes the adpative weights and proprocessed the X, y data (e.g. centering/scaling).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        X_pro, y_pro, sample_weight_pro, pro_pro_out, penalty_data

        X_pro: array-like, shape (n_samples, n_features)
            The processed covariate data.

        y_pro: array-like, shape (n_samples, )
            The processed response data.

        sample_weight_pro: None or array-like,  shape (n_samples,)
            The processed sample weights. Ensures sum(sample_weight) = n_samples. Possibly incorporate class weights.

        pro_pro_out: dict
            Data from preprocessing e.g. X_center, X_scale.

        penalty_data: dict
            A dict with key 'coef_init' containing the processed initial coefficient.

        """

        # get init data e.g. fit an initial estimator to the data
        init_data = self._get_init_data(X=X, y=y, sample_weight=sample_weight)

        # preproceess X, y
        X_pro, y_pro, sample_weight_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        # possibly add init_est to pre_pro_out so it gets saved in _set_fit
        if 'est' in init_data:
            pre_pro_out['init_est'] = init_data['est']

        # process init data
        init_data = process_init_data(init_data=init_data,
                                      pre_pro_out=pre_pro_out)

        return {'X_pro': X_pro,
                'y_pro': y_pro,
                'sample_weight_pro': sample_weight_pro,
                'pre_pro_out': pre_pro_out,

                'coef_init': init_data['coef'],
                'intercept_init': init_data['intercept']
                }

    def get_pen_val_max(self, X, y, sample_weight=None):
        """
        Returns the largest reasonable penalty parameter for the processed data.

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
        pen_val_max: float
            Largest reasonable tuning parameter value.
        """
        X_pro, y_pro, sample_weight_pro, _ = \
            self.preprocess(X, y, sample_weight=sample_weight, copy=True)

        return get_pen_max(X=X_pro, y=y_pro,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight_pro,
                           loss=self._get_loss_config(),
                           penalty=self._get_penalty_config()
                           )


class GlmNonConvexCV(GlmCV):

    def _get_estimator_for_cv(self, X, y=None, sample_weight=None):
        """
        Fits the initial coefficient and adds it to the 'init' argument.

        Output
        ------
        estimator
        """

        est = clone(self.estimator)

        # get initialization from raw data
        init_data = est._get_init_data(X, y, sample_weight=sample_weight)
        if 'est' in init_data:
            self.init_est_ = init_data['est']
            init_data.pop('est', None)

        # set initialization data
        est.set_params(init=init_data)

        return est
