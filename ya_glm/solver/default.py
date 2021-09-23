from ya_glm.solver.FISTA import FISTA
from ya_glm.solver.ZhuADMM import ZhuADMM

solvers_str2obj = {'fista': FISTA(),
                   'admm': ZhuADMM()}
avail_solvers = list(solvers_str2obj.keys())


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

        if solver == 'default':

            if loss.name == 'quantile' or \
                    (penalty is not None and
                        not penalty.is_smooth and
                        not penalty.is_proximable):

                # default to ADMM for non-smooth losses or penalties that
                # are neither smooth nor proximable
                return ZhuADMM()
            else:
                # TODO: return anderson CD when applicable
                return FISTA()

        else:
            if solver.lower() not in avail_solvers:
                raise ValueError("{} is not a valid solver string input, "
                                 "must be 'default' or one of {}".
                                 format(solver, avail_solvers))

            else:
                return solvers_str2obj[solver.lower()]
    else:
        return solver
