from copy import deepcopy

from ya_glm.GlmSolver import GlmSolver
from ya_glm.solver.cvxpy.glm_solver import solve_glm, solve_glm_path
from ya_glm.autoassign import autoassign


class CvxpySolver(GlmSolver):
    """
    Solves a penalized GLM problem using cvxpy.

    Parameters
    ---------
    zero_tol: float
        Values of the solution smaller than this are set to exactly zero.

    solver: None, str
        Which cvxpy solver to use. See cvxpy docs.

    cp_kws: dict
        Keyword arguments to the call to problem.solve(). See cvxpy docs.
    """

    @autoassign
    def __init__(self, zero_tol=1e-8, solver=None, cp_kws={}): pass

    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def _get_avail_losses(self):
        return ['lin_reg',  'log_reg', 'quantile']

    # TODO:
    def setup(self, X, y, loss, penalty, sample_weight=None):
        pass

    def solve(self, X, y, loss, penalty,
              fit_intercept=True,
              sample_weight=None,
              coef_init=None,
              intercept_init=None
              ):
        """
        Solves a penalized GLM problem. See docs for ya_glm.GlmSolver.
        """

        if loss.name not in self._get_avail_losses():
            raise ValueError("{} loss not available; this solver only"
                             "implements {}".format(loss.name,
                                                    self._get_avail_losses()))

        return solve_glm(X=X, y=y,
                         loss=loss,
                         fit_intercept=fit_intercept,
                         sample_weight=sample_weight,
                         coef_init=coef_init,
                         intercept_init=intercept_init,
                         **penalty.get_solve_kws(),
                         **self.get_solve_kws())

    def solve_path(self, X, y, loss, penalty_seq,
                   fit_intercept=True,
                   sample_weight=None,
                   coef_init=None,
                   intercept_init=None):
        """
        Solves a sequence of penalized GLM problems. See docs for ya_glm.GlmSolver.
        """

        if loss.name not in self._get_avail_losses():
            raise ValueError("{} loss not available; this solver only"
                             "implements {}".format(loss.name,
                                                    self._get_avail_losses()))

        if coef_init is not None or intercept_init is not None:
            # TODO do we want to allow this? perhaps warn?
            raise NotImplementedError

        return solve_glm_path(X=X, y=y,
                              loss=loss,
                              fit_intercept=fit_intercept,
                              sample_weight=sample_weight,
                              **penalty_seq.get_solve_kws(),
                              **self.get_solve_kws())

    def has_path_algo(self):
        """
        Yes this solver has an available path algorithm!
        """
        return True
