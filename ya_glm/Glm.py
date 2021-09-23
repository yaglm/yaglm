from ya_glm.base import BaseGlm
from ya_glm.LossMixin import LossMixin


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
            configs, solver, solver_init, inferencer = \
            self.setup_and_prefit(X, y, sample_weight)

        #########################
        # solve and set the fit #
        #########################
        self.inferencer_ = inferencer

        self._fit_from_configs(pro_data=pro_data, raw_data=raw_data,
                               configs=configs, solver=solver,
                               pre_pro_out=pre_pro_out,
                               solver_init=solver_init)

        return self
