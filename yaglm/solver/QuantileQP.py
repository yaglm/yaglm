from ya_glm.solver.base import GlmSolverWithPath
from ya_glm.autoassign import autoassign


class QuantileQP(GlmSolverWithPath):
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
                 verbosity=0):
        raise NotImplementedError("TODO need to finish re adding!")
