
# TODO: put coef_init, intercept_init in solve and solve_path?
class GlmSolver:

    def __init__(self): pass

    def setup(self, X, y, loss, penalty, sample_weight=None):
        pass

    def get_solve_kws(self):
        """
        Returns the optimization config parameters need to solve each GLM problem.

        Output
        ------
        kws: dict
            Any parameters from this config object that are used by self.solve.
        """
        raise NotImplementedError

    def solve(self, X, y, loss, penalty,
              fit_intercept=True,
              sample_weight=None,
              coef_init=None,
              intercept_init=None
              ):
        """
        Solves a penalized GLM problem.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        loss: ya_glm.loss.LossConfig.LossConfig
            A configuration object specifying the GLM loss.

        penalty: ya_glm.PenaltyConfig.PenaltyConfig
            A configuration object specifying the GLM penalty.

        fit_intercept: bool
            Whether or not to fit intercept, which is not penalized.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        coef_init: None, array-like
            (Optional) Initial value for the coefficient.

        intercept_init: None, array-like
            (Optional) Initial value for the intercept.

        Output
        ------
        coef, intercept, opt_data

        coef: array-like, shape (n_features, ) or (n_features, n_responses)
            The estimated coefficient.

        intercept: None, float or array-like, shape (n_responses, )
            The estimated intercept -- if one was requested.

        opt_data: dict
            Additional data returned by the optimizatoin algorithm.
        """
        raise NotImplementedError

    def solve_path(self, X, y, loss, penalty_seq,
                   fit_intercept=True,
                   sample_weight=None,
                   coef_init=None,
                   intercept_init=None):
        """"
        Solves the penalized GLM problem for a sequence of consecutive tuning values using a homotopy path algorithm. I.e. we compute the solution for one tuning value, then use that solution to warm start the algortihm for the next penalty value, etc.


        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        loss: ya_glm.loss.LossConfig.LossConfig
            A configuration object specifying the GLM loss.

        penalty_seq: ya_glm.PenaltyConfig.PenaltySequence
            A configuration object specifying the GLM penalty and tuning sequence.

        fit_intercept: bool
            Whether or not to fit intercept, which is not penalized.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        coef_init: None, array-like
            (Optional) Initial value for the coefficient.

        intercept_init: None, array-like
            (Optional) Initial value for the intercept.

        Output
        ------
        solution_path: iterator
            Each element of the is (fit_out, param_setting)

            fit_out: dict
                The soltuion.

            param_setting: dict
                The parameter setting using for this value of the path.
        """
        raise NotImplementedError

    def has_path_algo(self, loss, penalty):
        """
        Whether or not this solve has a path algorithm available for a given loss/penalty combination.
        """
        raise NotImplementedError
