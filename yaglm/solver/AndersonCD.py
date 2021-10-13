from yaglm.solver.base import GlmSolverWithPath
from yaglm.autoassign import autoassign


class AndersonCD(GlmSolverWithPath):
    """
    Solves a penalized GLM problem using the andersoncd package (https://github.com/mathurinm/andersoncd).

    Parameters
    ----------
    fake_intercept: True
        Andersoncd does not allow an intercept, but we can fake it in the same way sklearn.linear_model.Lasso does via appropriate centering.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.
    """

    @autoassign
    def __init__(self, max_iter=20, max_epochs=50000,
                 p0=10, verbose=0, tol=1e-4, prune=0):
        raise NotImplementedError("TODO need to finish re adding!")
