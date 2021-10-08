from yaglm.solver.base import GlmSolver
from yaglm.autoassign import autoassign


class QuantileLP(GlmSolver):
    """
    Solves a penalized quantile regression problem with a Linear Program formulation using a scipy backend.

    Parameters
    ----------
    lp_solver: str
        Which scipy linear program solver to use.
        See scipy.optimize.linprog.

    lp_kws: dict
        Keyword arguments for yaglm.solver.quantile_lp.scipy_lin_prog.solve

    verbosity: int
        How much printout do we want.
    """
    @autoassign
    def __init__(self, slp_solver='highs',
                 lp_kws={},
                 verbosity=0):
        raise NotImplementedError("TODO need to finish re adding!")
