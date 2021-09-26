from ya_glm.solver.FISTA import FISTA
from ya_glm.solver.ZhuADMM import ZhuADMM




def get_solver(solver, loss, penalty=None, constraint=None):
    """
    Parameters
    ----------
    solver: str, GlmSolver

    loss: LossConfig

    penalty: None, PenaltyConfig

    constraint: None, ConstraintConfig

    Output
    ------
    solver: GlmSolver
    """

    if isinstance(solver, str):

        if solver == 'default':  # return default solver

            # use FISTA by default if it is applicable
            if FISTA.is_applicable(loss=loss,
                                   penalty=penalty,
                                   constraint=constraint):

                return FISTA()
            else:
                return ZhuADMM()

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
                   'admm': ZhuADMM()}
avail_solvers = list(solvers_str2obj.keys())
