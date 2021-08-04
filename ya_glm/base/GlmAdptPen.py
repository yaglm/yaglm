import numpy as np
from sklearn.base import clone

from ya_glm.base.GlmCV import GlmCV
from ya_glm.base.GlmCvxPen import GlmCvxPen
from ya_glm.base.InitFitMixin import InitFitMixin

from ya_glm.processing import process_init_data
from ya_glm.pen_max.lasso import get_pen_max
from ya_glm.pen_max.ridge import get_ridge_pen_max

# TODO: save adpat_weights somewhere e.g. perhaps put in pre_pro_out


class GlmAdptPen(InitFitMixin, GlmCvxPen):
    """
    Base class for adpative penalties.

    Parameters
    ----------
    adpt_func: str
        The concave function whose gradient is used to obtain the adpative weights from the initial coefficient. See ya_glm.opt.penalty.concave_penalty.

    adpt_func_kws: dict
        Keyword arguments to the adpative function e.g. q for the Lq norm.

    pertub_init: str, float
        How to perturb the initial coefficient i.e. we evaluate the adpative function's gradient at abs(init_coef) + pertub_init. If pertub_init='n_samples', then we use 1/n_samples. This perturbation is useful, for example, when the init coefficient has exact zeros and adpt_func='log'.
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
            A dict with keys either 'adpt_weights' (if these were provided to 'init') or  'coef_init', 'n_samples' whose values are used to compute the adpative weights.

        """

        penalty_data = {}

        # get init data e.g. fit an initial estimator to the data
        init_data = self._get_init_data(X=X, y=y, sample_weight=sample_weight)

        if 'est' in init_data:
            penalty_data['init_est'] = init_data['est']

        # preproceess X, y
        X_pro, y_pro, sample_weight_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        if self.has_preset_adpt_weights():
            # if the adpative weights have been provided then just return them
            penalty_data['adpt_weights'] = init_data['adpt_weights']

        else:  # otherwise get info for computing adpative weights
            # process init data
            init_data = process_init_data(init_data=init_data,
                                          pre_pro_out=pre_pro_out)
            penalty_data['coef_init'] = np.array(init_data['coef'])
            penalty_data['n_samples'] = X.shape[0]

        return {'X_pro': X_pro,
                'y_pro': y_pro,
                'sample_weight_pro': sample_weight_pro,
                'pre_pro_out': pre_pro_out,

                'penalty_data': penalty_data
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
        X_pro, y_pro, sample_weight_pro, pre_pro_out, penalty_data = \
            self.prefit(X=X, y=y, sample_weight=sample_weight)

        # tell the penalty about the adpative weights
        penalty = self._get_penalty_config()
        penalty.set_data(penalty_data)

        if self._primary_penalty_type == 'lasso':

            return get_pen_max(X=X_pro, y=y_pro,
                               fit_intercept=self.fit_intercept,
                               sample_weight=sample_weight_pro,
                               loss=self._get_loss_config(),
                               penalty=penalty.cvx_pen
                               )

        elif self._primary_penalty_type == 'ridge':
            return get_ridge_pen_max(X=X_pro, y=y_pro,
                                     fit_intercept=self.fit_intercept,
                                     sample_weight=sample_weight_pro,
                                     loss=self._get_loss_config(),
                                     penalty=penalty.cvx_pen
                                     )

        else:
            raise ValueError("Bad self._primary_penalty_type: {}"
                             "".format(self._primary_penalty_type))

    def has_preset_adpt_weights(self):
        if isinstance(self.init, dict) and 'adpt_weights' in self.init.keys():
            return True
        else:
            return False


class GlmAdptPenCV(GlmCV):
    """
    Base class for cross-validating adpative penalties.
    """

    def _get_estimator_for_cv(self, X, y=None, sample_weight=None):
        """
        Fits the initial coefficient, computes the adpative weights and adds 'adpt_weights' to the 'init' argument.

        Output
        ------
        estimator
        """
        est = clone(self.estimator)

        # if the estimator has preset adaptive weights the just
        # return the estimator
        if est.has_preset_adpt_weights():
            return est

        # get initialization from raw data
        init_data = est._get_init_data(X, y, sample_weight=sample_weight)
        if 'est' in init_data:
            self.init_est_ = init_data['est']
            init_data.pop('est', None)

        # process init coef
        X_pro, y_pro, sample_weight_pro, pre_pro_out = \
            est.preprocess(X=X, y=y,
                           sample_weight=sample_weight,
                           copy=True,
                           check_input=False)

        init_data = process_init_data(init_data=init_data,
                                      pre_pro_out=pre_pro_out)
        coef_init = np.array(init_data['coef'])

        # compute adpative weights
        config = est._get_penalty_config()
        adpt_weights = config.compute_adpative_weights(coef_init=coef_init,
                                                       n_samples=X_pro.shape[0])

        # set adpative weights in init data
        init_data['adpt_weights'] = adpt_weights
        est.set_params(init=init_data)

        return est
