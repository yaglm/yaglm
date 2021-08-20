from ya_glm.GlmSolver import GlmSolver
from ya_glm.autoassign import autoassign

from ya_glm.solver.quantile_lp.scipy_lin_prog import solve as solve_lin_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve as solve_quad_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve_path

from warnings import warn


class QuantileLProgSolver(GlmSolver):
    """
    Solves a penalized quantile regression problem with a Linear Program formulation using a scipy backend.

    Parameters
    ----------
    lp_solver: str
        Which scipy linear program solver to use.
        See scipy.optimize.linprog.

    lp_kws: dict
        Keyword arguments for ya_glm.solver.quantile_lp.scipy_lin_prog.solve

    verbosity: int
        How much printout do we want.
    """

    @autoassign
    def __init__(self, slp_solver='highs',
                 lp_kws={},
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

        if penalty.ridge:
            raise NotImplementedError("The LP solver does not support ridge penalties")

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

        kws.update({'lasso_pen_val': penalty.lasso_pen_val,
                    'lasso_weights': penalty.lasso_weights,
                    })

        kws.pop('tikhonov', None)
        kws.pop('ridge_pen_val', None)
        kws.pop('ridge_weights', None)

        return solve_lin_prog(solver=self.scipy_lp_solver,
                              **self.scipy_lp_kws,
                              **kws)

    def has_path_algo(self):
        return False


class QuantileCvxpyLQProgSolver(GlmSolver):
    """
    Solves a penalized quantile regression problem with either a Linear Program or Quadratic program formulation using a cvxpy backend.

    Parameters
    ----------
    solver: str
        Which cvxpy solver to use for quadratic programs.

    solver_kws: dict
        Keyword arguments to the call to cvxpy's solve

    verbosity: int
        How much printout do we want.
    """

    @autoassign
    def __init__(self, solver='ECOS', solver_kws={},
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

        return solve_quad_prog(solver=self.solver,
                               cp_kws=self.solver_kws,
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

        return solve_path(solver=self.solver,
                          cp_kws=self.solver_kws,
                          **kws,
                          **penalty_seq.get_solve_kws())

    def has_path_algo(self):
        return True
