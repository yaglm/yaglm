from ya_glm.base import BaseGlm, initialize_configs
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

        ###########################
        # preprocessing and setup #
        ###########################

        # basic formatting check
        X, y, sample_weight = self._validate_data(X=X, y=y,
                                                  sample_weight=sample_weight)

        # run any prefitting inference
        inferencer = self.run_prefit_inference(X, y,
                                               sample_weight=sample_weight)

        # preproceess X, y
        X_pro, y_pro, sample_weight_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        # possibly fit initializer estimator for flavored penalties
        # that need one
        init_data_pro, init_est = \
            self.get_initializer(X=X, y=y,
                                 pre_pro_out=pre_pro_out,
                                 sample_weight=sample_weight)

        # store init_est here to be saved later in _pre_fit
        pre_pro_out['init_est'] = init_est

        # initialize config objects
        configs, solver, solver_init = \
            initialize_configs(solver=self.solver,
                               loss=self.loss,
                               penalty=self.penalty,
                               constraint=self.constraint,
                               init_data=init_data_pro)

        #########################
        # solve and set the fit #
        #########################
        self.inferencer_ = inferencer
        self._fit_from_configs(configs=configs,
                               solver=solver,
                               pre_pro_out=pre_pro_out,
                               solver_init=solver_init,
                               X=X, y=y, X_pro=X_pro, y_pro=y_pro,
                               sample_weight=sample_weight,
                               sample_weight_pro=sample_weight_pro)

        return self
