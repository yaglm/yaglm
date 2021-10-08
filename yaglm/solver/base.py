from copy import deepcopy

from yaglm.config.base import Config

from yaglm.config.loss import get_loss_config
from yaglm.config.constraint import get_constraint_config
from yaglm.config.penalty import get_penalty_config
from yaglm.config.base_params import get_base_config, detune_config
from yaglm.config.penalty_utils import get_unflavored


class GlmSolver(Config):

    def __init__(self): pass

    @classmethod
    def is_applicable(self, loss, penalty=None, constraint=None, lla=False):
        """
        Determines whether or not this solver is applicable to a given optimization problem.

        Parameters
        ----------
        loss: LossConfig
            The loss.

        penalty: None, PenaltyConfig
            The penalty.

        constraint: None, ConstraintConfig
            The constraint

        lla: bool
            Whether or not this solver will be used to solver LLA subproblems i.e. if it will only see convex versions of a non-convex penalty.

        Output
        ------
        is_applicable: bool
            Whether or not this solver can be used.
        """
        # pull out base configs
        loss = get_base_config(get_loss_config(loss))
        penalty = get_base_config(get_penalty_config(penalty))
        if constraint is not None:
            constraint = get_base_config(get_constraint_config(constraint))

        # the LLA algorithm only sees unflavored versions of the penalty
        _penalty = detune_config(deepcopy(penalty))
        if lla:
            _penalty = get_unflavored(_penalty)

        return self._is_applicable(loss=loss, penalty=_penalty,
                                   constraint=constraint)

    @classmethod
    def _is_applicable(self, loss, penalty=None, constraint=None):
        raise NotImplementedError("Sub-class should overwrite")

    def get_solve_kws(self):
        """
        Returns the optimization config parameters need to solve each GLM problem.

        Output
        ------
        kws: dict
            Any parameters from this config object that are used by self.solve.
        """
        return self.get_params()

    def setup(self, X, y, loss, penalty, constraint=None,
              fit_intercept=True, sample_weight=None):
        """
        Sets up anything the solver needs.
        """
        raise NotImplementedError

    def update_penalty(self, **params):
        """
        Updates the penalty value.
        """
        raise NotImplementedError

    def solve(self, coef_init=None, intercept_init=None, other_init=None):
        """
        Solves the optimization problem.

        Parameters
        ----------
        coef_init: None, array-like
            (Optional) Initialization for the coefficient.

        intercept_init: None, array-like
            (Optional) Initialization for the intercept.

        other_init: None, array-like
            (Optional) Initialization for other optimization data e.g. dual variables.

        Output
        ------
        soln, other_data, opt_info

        soln: dict of array-like
            The coefficient/intercept solutions,

        other_data: dict
            Other optimzation output data e.g. dual variables.

        opt_info: dict
            Optimization information e.g. number of iterations, runtime, etc.
        """
        raise NotImplementedError

    @property
    def has_path_algo(self):
        """
        Whether or not this solve has a path algorithm available for a given loss/penalty combination.
        """
        return False

    @property
    def needs_fixed_init(self):
        """
        Whether or not this solver has a fixed initializer.
        """
        return False

    def set_fixed_init(self):
        raise NotImplementedError('Subclass should overwrite')


class GlmSolverWithPath(GlmSolver):

    def solve_penalty_path(self, penalty_path,
                           coef_init=None,
                           intercept_init=None,
                           other_init=None):
        """
        Solves the optimization problem over a penalty parameter path using warm starts.

        Parameters
        ----------
        penalty_path: iterable
            Iterates over the penalty path parameters.

        coef_init: None, array-like
            (Optional) Initialization for the coefficient.

        intercept_init: None, array-like
            (Optional) Initialization for the intercept.

        other_init: None, array-like
            (Optional) Initialization for other optimization data e.g. dual variables.

        Yields
        ------
        soln, other_data, opt_info

        soln: dict of array-like
            The coefficient/intercept solutions,

        other_data: dict
            Other optimzation output data e.g. dual variables.

        opt_info: dict
            Optimization information e.g. number of iterations, runtime, etc.
        """

        for path_val_dict in penalty_path:
            self.update_penalty(**path_val_dict)

            soln, opt_data, opt_info = self.solve(coef_init=coef_init,
                                                  intercept_init=intercept_init,
                                                  other_init=other_init)

            yield soln, opt_data, opt_info

            # update for warm start
            coef_init = soln['coef']
            intercept_init = soln['intercept']
            other_init = opt_data

    @property
    def has_path_algo(self):
        """
        Whether or not this solve has a path algorithm available for a given loss/penalty combination.
        """
        return True
