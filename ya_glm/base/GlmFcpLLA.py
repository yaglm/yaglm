from sklearn.base import clone
import numpy as np

from ya_glm.base.Glm import Glm
from ya_glm.base.GlmCV import GlmCV
from ya_glm.base.InitFitMixin import InitFitMixin

from ya_glm.lla.LLASolver import LLASolver

from ya_glm.solver.default import get_default_solver
from ya_glm.processing import process_init_data
from ya_glm.pen_max.fcp_lla import get_pen_max


class GlmFcpLLA(InitFitMixin, Glm):
    """
    Base class for folded concave penalties fit with the LLA algorithm.

    Parameters
    ----------
    pen_val: float
        The penalty value for the concave penalty.

    pen_func: str
        The concave penalty function. See ya_glm.opt.penalty.concave_penalty.

    pen_func_kws: dict
        Keyword arguments for the concave penalty function e.g. 'a' for the SCAD function.

    lla_n_steps: int
        Maximum of steps the LLA algorithm should take. The LLA algorithm can have favorable statistical properties after only 1 step.

    lla_kws: dict
        Additional keyword arguments to the LLA algorithm solver excluding 'n_steps' and 'glm_solver'. See ya_glm.lla.LLASolver.LLASolver.
    """
    def _get_glm_solver(self):
        """
        Returns the glm solver used for the LLA subproblems.

        Output
        ------
        solver: ya_glm.GlmSolver
            The solver config object.
        """
        if type(self.glm_solver) == str and self.glm_solver == 'default':
            return get_default_solver(loss=self._get_loss_config(),
                                      penalty=self._get_penalty_config())

        else:
            return self.glm_solver

    def _get_solver(self):
        """
        Returns the LLA solver.

        Output
        ------
        solver: ya_glm.lla.LLASolver.LLASolver
            The LLA solver config object.
        """
        return LLASolver(glm_solver=self._get_glm_solver(),
                         n_steps=self.lla_n_steps,
                         **self.lla_kws)

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
        X_pro: array-like, shape (n_samples, n_features)
            The processed covariate data.

        y_pro: array-like, shape (n_samples, )
            The processed response data.

        pro_pro_out: dict
            Data from preprocessing e.g. X_center, X_scale.

        penalty_data: dict
            A dict with key 'coef_init' containing the processed initial coefficient.

        """

        penalty_data = {}

        # get init data e.g. fit an initial estimator to the data
        init_data = self._get_init_data(X=X, y=y, sample_weight=sample_weight)

        if 'est' in init_data:
            penalty_data['init_est'] = init_data['est']

        # preproceess X, y
        X_pro, y_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        # process init data
        init_data = process_init_data(init_data=init_data,
                                      pre_pro_out=pre_pro_out)
        penalty_data['coef_init'] = np.array(init_data['coef'])

        return X_pro, y_pro, pre_pro_out, penalty_data

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

        # get initial coefficient
        X_pro, y_pro, _, penalty_data = \
            self.prefit(X=X, y=y, sample_weight=sample_weight)

        # tell the penalty config about the initial coefficient
        penalty = self._get_penalty_config()
        penalty.set_data(penalty_data)

        return get_pen_max(X=X, y=y,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight,
                           loss=self._get_loss_config(),
                           penalty=penalty
                           )


class GlmFcpLLACV(GlmCV):

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
