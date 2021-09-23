from ya_glm.solver.FISTA import FISTA
from ya_glm.solver.ZhuADMM import ZhuADMM
from ya_glm.solver.LLA import LLAFixedInit
from ya_glm.config.utils import is_flavored
from ya_glm.config.base import safe_get_config


solvers_str2obj = {'fista': FISTA()}
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


def needs_lla_solver(penalty):
    """
    Checks if we need an LLA solver.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty config. Only needs the LLA algorithm if this is a non-convex lla flavored penalty.

    Output
    ------
    needs_lla: bool
    """
    if is_flavored(penalty) and \
            safe_get_config(penalty.flavor).name == 'non_convex_lla':
        return True
    else:
        return False


def maybe_get_lla(penalty, glm_solver):
    """
    Wraps a base glm solver in the LLA algorithm if it is required.

    Parameters
    ----------
    penalty: PenaltyConfig
        The penalty config. Only needs the LLA algorithm if this is a non-convex lla flavored penalty.

    glm_solver: GlmSolver
        The base GLM solver.

    Output
    ------
    solver: GlmSolver
        Either the LLA solver or the original glm_solver.
    """
    if needs_lla_solver(penalty):
        lla_solver = LLAFixedInit(**safe_get_config(penalty.flavor).\
                                  lla_solver_kws)
        lla_solver.set_sp_solver(glm_solver)
        return lla_solver

    else:
        return glm_solver
