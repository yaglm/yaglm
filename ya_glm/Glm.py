from copy import deepcopy

from ya_glm.base import BaseGlm
from ya_glm.LossMixin import LossMixin
from ya_glm.GlmTuned import GlmCV
from ya_glm.config.base_penalty import get_unflavored, get_flavor_info
from ya_glm.config.base_params import get_base_config
from ya_glm.adaptive import set_adaptive_weights


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

        ##############################################
        # setup, preprocess, and prefitting routines #
        ##############################################
        pro_data, raw_data, pre_pro_out, \
            configs, solver, init_data, inferencer = \
            self.setup_and_prefit(X, y, sample_weight)

        ########################
        # Set adaptive weights #
        ########################

        penalty = configs['penalty']
        if get_flavor_info(penalty) == 'adaptive':
            base_penalty = get_base_config(penalty)

            # set the adaptive weights in place
            set_adaptive_weights(penalty=base_penalty,
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
