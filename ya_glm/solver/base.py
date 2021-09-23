from ya_glm.config.base import _Config


class GlmSolver(_Config):

    def __init__(self): pass

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
        Updates the penalty parameters.
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
