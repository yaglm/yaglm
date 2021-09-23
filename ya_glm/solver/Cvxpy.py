from ya_glm.solver.base import GlmSolverWithPath
from ya_glm.autoassign import autoassign


class Cvxpy(GlmSolverWithPath):
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
    def __init__(self, zero_tol=1e-8, solver=None, cp_kws={}):
        raise NotImplementedError("TODO need to finish re adding!")
