from copy import deepcopy

from yaglm.base import BaseGlm
from yaglm.LossMixin import LossMixin
from yaglm.GlmTuned import GlmCV
from yaglm.config.penalty_utils import get_unflavored, get_flavor_kind
from yaglm.adaptive import set_adaptive_weights


class Glm(LossMixin, BaseGlm):

    @property
    def _is_tuner(self):
        return False

    def fit(self, X, y, sample_weight=None):
        """
        Fits the penalized GLM.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            (Optional) Individual weights for each sample.

        Output
        ------
        self
            Fitted estimator.
        """

        # TODO: make sure none of the configs are tuners

        ##############################################
        # setup, preprocess, and prefitting routines #
        ##############################################
        pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer = \
            self.setup_and_prefit(X, y, sample_weight)

        ########################
        # Set adaptive weights #
        ########################

        if get_flavor_kind(configs['penalty']) in ['adaptive', 'mixed']:
            configs['penalty'] = \
                set_adaptive_weights(penalty=configs['penalty'],
                                     init_data=init_data)

        #########################
        # solve and set the fit #
        #########################
        self.inferencer_ = inferencer

        self._fit_from_configs(pro_data=pro_data, raw_data=raw_data,
                               configs=configs, solver=solver,
                               pre_pro_out=pre_pro_out,
                               init_data=init_data)

        return self

    def get_unflavored_tunable(self):
        """
        Gets a cross-validation version of this estimator. Ensures the penalty is not flavored.

        Output
        ------
        est: GlmCV
            A cross-validation estimator.
        """
        params = deepcopy(self.get_params(deep=False))
        params['initializer'] = 'default'  # no need for initializer

        # unflavor penalty
        params['penalty'] = get_unflavored(params['penalty'])

        return GlmCV(**params)
