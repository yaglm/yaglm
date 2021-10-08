from yaglm.solver.FISTA import FISTA
from yaglm.solver.ZhuADMM import ZhuADMM
from yaglm.solver.Cvxpy import Cvxpy


def get_solver(solver='default', loss='lin_reg',
               penalty=None, constraint=None, lla=False):
    """
    Returns a GlmSolver object.

    Parameters
    ----------
    solver: str, GlmSolver
        The solver we want. If 'default', we will try to guess the best solver to use.

    loss: LossConfig
        The loss config.

    penalty: None, PenaltyConfig
        The penalty config.

    constraint: None, ConstraintConfig
        The constraint config.

    lla: bool
        Whether or not this solver will be used for LLA subproblems, which are convex.

    Output
    ------
    solver: GlmSolver
    """

    if isinstance(solver, str):

        # return default solver
        # current priority: fista, cvxpy, admm
        # currently our ADMM is not consistenly better than cvxpy
        if solver == 'default':

            # use FISTA by default if it is applicable
            if FISTA.is_applicable(loss=loss,
                                   penalty=penalty,
                                   constraint=constraint,
                                   lla=lla):

                return FISTA()

            # only import cvxpy if we need to!

            if Cvxpy.is_applicable(loss=loss,
                                   penalty=penalty,
                                   constraint=constraint,
                                   lla=lla):
                return Cvxpy()

            # back up with ZhuADMM
            elif ZhuADMM.is_applicable(loss=loss,
                                       penalty=penalty,
                                       constraint=constraint,
                                       lla=lla):
                return ZhuADMM()

            else:
                raise ValueError("No available solver found for "
                                 "loss={}, penalty={}, constrain={}".
                                 format(loss, penalty, constraint))

        else:  # return user specified solver
            if solver.lower() not in avail_solvers:
                raise ValueError("{} is not a valid solver string input, "
                                 "must be 'default' or one of {}".
                                 format(solver, avail_solvers))

            else:
                return solvers_str2obj[solver.lower()]
    else:
        return solver


solvers_str2obj = {'fista': FISTA(),
                   'admm': ZhuADMM(),
                   'cvxpy': Cvxpy()
                   }
avail_solvers = list(solvers_str2obj.keys())
