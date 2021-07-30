import numpy as np

from ya_glm.lla.utils import safe_concat
from ya_glm.opt.glm_loss.get import get_glm_loss


class BaseWeightedLassoSolver(object):
    """
    Based class for solving weighted Lasso-like subproblems for the LLA algorithm.
    """

    def solve(self, L1_weights, opt_init=None, opt_init_upv=None):
        """
        Parameters
        ----------
        L1_weights: array-like
            Weights for lasso penalty.

        opt_init: None, array-like
            Optional initialization for the penalized variable.

        opt_init_upv: None, array-like
            Optional initialization for the un-penalized variable.

        Output
        ------
        solution, upv_solution, other_data
        """
        raise NotImplementedError

    def loss(self, value, upv=None):
        """
        Returns the loss function

        loss(y) or loss(y, u)

        Parameters
        ----------
        value: array-like
            Value of the variable.

        upv: None, array-like
            Optional unpenalized variable.

        Output
        ------
        loss: float
        """
        raise NotImplementedError


class WeightedLassoGlmSolver(BaseWeightedLassoSolver):
    """
    Solver for GLMs with weighted Lasso-like penalites.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The response data.

    solver: ya_glm.GlmSolver
        The solver to use for the penalized GLM subproblems.

    loss: ya_glm.loss.LossConfig.LossConfig
        A configuration object specifying the GLM loss.

    base_penalty: ya_glm.PenaltyConfig.PenaltyConfig
        A configuration object specifying GLM subproblem penalty (excluding the Lasso weights that will be provided).

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    """

    def __init__(self, X, y, solver, loss, base_penalty, fit_intercept=True,
                 sample_weight=None):

        self.solver = solver

        self.X = X
        self.y = y
        self.loss = loss

        # base lasso subproblem penalty
        self.base_penalty = base_penalty
        self.base_penalty.lasso_pen_val = 1
        self.base_penalty.lasso_weights = np.empty(0)

        self.fit_intercept = fit_intercept
        self.sample_weight = sample_weight

        self.glm_loss = get_glm_loss(X=X, y=y, loss=loss,
                                     fit_intercept=fit_intercept,
                                     sample_weight=sample_weight)

    def solve(self, L1_weights, opt_init=None, opt_init_upv=None):
        """
        Parameters
        ----------
        L1_weights: array-like
            Weights for lasso penalty.

        opt_init: None, array-like
            Optional initializaiton for the coefficient.

        opt_init_upv: None, array-like
            Optional initializaiton for the intercept.

        Output
        ------
        solution, upv_solution, other_data
        """

        self.base_penalty.lasso_weights = L1_weights

        return self.solver.solve(X=self.X,
                                 y=self.y,
                                 loss=self.loss,
                                 penalty=self.base_penalty,
                                 fit_intercept=self.fit_intercept,
                                 sample_weight=self.sample_weight,
                                 coef_init=opt_init,
                                 intercept_init=opt_init_upv
                                 )

    def get_loss(self, value, upv=None):
        """
        Returns the loss function

        loss(y) or loss(y, u)

        Parameters
        ----------
        value: array-like
            The value of the coefficient.

        upv: None, array-like
            The intercept.

        Output
        ------
        loss: float
        """
        if self.glm_loss.fit_intercept:
            # return self.glm_loss.eval(np.concatenate([[upv], value]))
            return self.glm_loss.eval(safe_concat(upv, value))
        else:
            return self.glm_loss.eval(value)
