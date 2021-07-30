from ya_glm.GlmSolver import GlmSolver
from ya_glm.autoassign import autoassign

from ya_glm.solver.quantile_lp.scipy_lin_prog import solve as solve_lin_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve as solve_quad_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve_path

from warnings import warn


class QuantileLQProgSolver(GlmSolver):
    """
    Solves a penalized quantile regression problem using either a Linear Program or Quadratic program formulation.

    Parameters
    ----------
    scipy_lp_solver: str
        Which scipy linear program solver to use.
        See scipy.optimize.linprog.

    scipy_lp_kws: dict
        Keyword arguments for ya_glm.solver.quantile_lp.scipy_lin_prog.solve

    cvxpy_solver: str
        Which cvxpy solver to use for quadratic programs.

    cvxpy_solver_kws: dict
        Keyword arguments to the call to cvxpy's solve

    verbosity: int
        How much printout do we want.
    """

    @autoassign
    def __init__(self, scipy_lp_solver='highs',
                 scipy_lp_kws={},
                 cvxpy_solver='ECOS',
                 cvxpy_solver_kws={},
                 verbosity=0): pass

    def setup(self, X, y, loss, penalty, sample_weight=None):
        pass

    def solve(self, X, y, loss, penalty,
              fit_intercept=True,
              sample_weight=None,
              coef_init=None,
              intercept_init=None
              ):
        """
        Solves a quantile regression linear program using scipy or a quantile regression quadratic progam using cvxpy. See docs for ya_glm.GlmSolver.
        """

        if loss.name != 'quantile':
            raise ValueError("This solver only implements quantile regression")

        if penalty.get_penalty_kind() != 'entrywise':
            raise NotImplementedError("This solver only works for entrywise penalties")

        if y.ndim == 2 and y.shape[1] > 1:
            raise NotImplementedError("This solver currently only supports one dimensional responses")

        if self.verbosity >= 1:
            if coef_init is not None or intercept_init is not None:
                warn("coef_init and intercept_init not used")

        kws = {'X': X,
               'y': y,
               'fit_intercept': fit_intercept,
               'sample_weight': sample_weight,
               'quantile': loss.quantile
               }

        # add penalty keyword args to kws
        pen_kws = penalty.get_solve_kws()
        for k in ['groups', 'nuc', 'multi_task']:
            pen_kws.pop(k, None)

        kws.update(pen_kws)

        if penalty.ridge_pen_val is None:
            kws.update({'lasso_pen_val': penalty.lasso_pen_val,
                        'lasso_weights': penalty.lasso_weights,
                        })

            kws.pop('tikhonov', None)
            kws.pop('ridge_pen_val', None)
            kws.pop('ridge_weights', None)

            return solve_lin_prog(solver=self.scipy_lp_solver,
                                  **self.scipy_lp_kws,
                                  **kws)

        else:

            return solve_quad_prog(solver=self.cvxpy_solver,
                                   cp_kws=self.cvxpy_solver_kws,
                                   **kws)

    def solve_path(self, X, y, loss, penalty_seq,
                   fit_intercept=True,
                   sample_weight=None,
                   coef_init=None,
                   intercept_init=None):

        """
        Solves a sequence of penalized quantile regression problems that are formulated as quadratic programs and solved using cvxpy. See docs for ya_glm.GlmSolver.
        """

        if loss.name != 'quantile':
            raise ValueError("This solver only implements quantile regression")

        if loss.name != 'quantile':
            raise ValueError("This solver only implements quantile regression")

        if coef_init is not None or intercept_init is not None:
            # TODO do we want to allow this? perhaps warn?
            raise NotImplementedError
        # if self.verbosity >= 1:
        #     if coef_init is not None or intercept_init is not None:
        #         warn("coef_init and intercept_init not used")

        kws = {'X': X,
               'y': y,
               'fit_intercept': fit_intercept,
               'sample_weight': sample_weight,
               'quantile': loss.quantile
               }

        return solve_path(solver=self.cvxpy_solver,
                          cp_kws=self.cvxpy_solver_kws,
                          **kws,
                          **penalty_seq.get_solve_kws())

    def has_path_algo(self, loss, penalty):
        """
        Currently only cvxpy supports path algorithms
        """

        # no path algorithm for the linear program version
        if penalty.ridge is None:
            return False
        else:
            # there is a path algo for ridge
            return True
