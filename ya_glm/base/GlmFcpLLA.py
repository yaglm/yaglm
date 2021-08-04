from ya_glm.base.GlmNonConvex import GlmNonConvex
from ya_glm.lla.LLASolver import LLASolver

from ya_glm.solver.default import get_default_solver
from ya_glm.pen_max.fcp_lla import get_pen_max


class GlmFcpLLA(GlmNonConvex):
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
        X_pro, y_pro, sample_weight_pro, pre_pro_out, penalty_data = \
            self.prefit(X=X, y=y, sample_weight=sample_weight)

        # tell the penalty config about the initial coefficient
        penalty = self._get_penalty_config()
        penalty.set_data(penalty_data)

        return get_pen_max(X=X, y=y,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight_pro,
                           loss=self._get_loss_config(),
                           penalty=penalty
                           )
