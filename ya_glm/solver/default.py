from ya_glm.solver.FISTA import FISTA
from ya_glm.solver.ZhuADMM import ZhuADMM

from ya_glm.config.loss import get_loss_config
from ya_glm.config.constraint import get_constraint_config
from ya_glm.config.penalty import get_penalty_config
from ya_glm.config.base_params import get_base_config


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

    # pull out base config objects
    loss = get_base_config(get_loss_config(loss))
    penalty = get_base_config(get_penalty_config(penalty))
    constraint = get_base_config(get_constraint_config(constraint))

    if isinstance(solver, str):

        if solver == 'default':  # return default solver

            # extract penalty function information
            if penalty is not None:
                pen_info = penalty.get_func_info()

            # use FISTA for smooth loss + proximable penalty;
            # otherwise fall back on ADMM
            if loss.name == 'quantile' or \
                    (penalty is not None and
                        not pen_info['smooth'] and
                        not pen_info['proximable']):

                # default to ADMM for non-smooth losses or penalties that
                # are neither smooth nor proximable
                return ZhuADMM()
            else:
                return FISTA()

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
