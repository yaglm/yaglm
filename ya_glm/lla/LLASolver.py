from copy import deepcopy

from ya_glm.GlmSolver import GlmSolver
from ya_glm.lla.lla import solve_lla
from ya_glm.lla.WeightedLassoSolver import WeightedLassoGlmSolver

from ya_glm.autoassign import autoassign
from ya_glm.PenaltyConfig import get_convex_base_from_concave


class LLASolver(GlmSolver):
    """
    Solves a concave penalized GLM problem using the LLA algorithm.

    Parameters
    ----------
    n_steps: int
        Number of LLA steps to take.

    xtol: float, None
        The change in x tolerance stopping criterion based on the L_infy norm.

    atol: float, None
        Absolute tolerance for loss based stopping criterion.

    rtol: float, None
        Relative tolerance for loss based stopping criterion.

    tracking_level: int
        How much optimization data to store at each step. Lower values means less informationed is stored.

    verbosity: int
        How much information to print out. Lower values means less print out.

    glm_solver: ya_glm.GlmSolver
        The solver to use for the penalized GLM subproblems.
    """

    @autoassign
    def __init__(self, n_steps=1, xtol=1e-4, atol=None, rtol=None,
                 tracking_level=1, verbosity=0,
                 glm_solver=None): pass

    def get_solve_kws(self):
        kws = deepcopy(self.__dict__)
        kws.pop('glm_solver')
        return kws

    def setup(self, X, y, loss, penalty, sample_weight=None):
        self.glm_solver.setup(X=X, y=y, loss=loss, penalty=penalty,
                              sample_weight=sample_weight)

    def solve(self, X, y, loss, penalty, coef_init,
              fit_intercept=True,
              sample_weight=None,
              intercept_init=None
              ):
        """
        Solves a penalized GLM problem using the LLA algorithm. See docs for ya_glm.GlmSolver.
        """

        # get base Lasso penalty for subproblems
        convex_base = get_convex_base_from_concave(penalty)

        # setup weighted lasso subproblem solver
        wlasso_solver = WeightedLassoGlmSolver(X=X, y=y,
                                               fit_intercept=fit_intercept,
                                               sample_weight=sample_weight,
                                               loss=loss,
                                               base_penalty=convex_base,
                                               solver=self.glm_solver
                                               )

        # get initializer for intercept -- the solve_lla needs and intercept
        # initializer if an intercept is present
        if fit_intercept and intercept_init is None:
            intercept_init = wlasso_solver.glm_loss.intercept_at_coef_eq0()

        return solve_lla(wlasso_solver=wlasso_solver,
                         penalty_fcn=penalty.get_penalty_func(),
                         transform=penalty._get_coef_transform(),
                         init=coef_init,
                         init_upv=intercept_init,
                         **self.get_solve_kws()
                         )

    # TODO: perhaps implement this?
    def has_path_algo(self, loss, penalty):
        return False
