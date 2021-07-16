from ya_glm.lla.utils import safe_concat
from ya_glm.backends.fista.glm_solver import get_glm_loss


class BaseWeightedLassoSolver(object):
    """
    min_y loss(y) + ||y||_{w, 1}

    or

    min_{y, u} loss(y, u) + ||y||_{w, 1}
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


class WL1SolverGlm(BaseWeightedLassoSolver):

    solve_glm = None

    def __init__(self, X, y, loss_func, loss_kws,
                 fit_intercept, solver_kws={}):

        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.loss_kws = loss_kws
        self.fit_intercept = fit_intercept
        self.solver_kws = solver_kws

        self.glm_loss = get_glm_loss(X=X, y=y,
                                     loss_func=loss_func,
                                     loss_kws=loss_kws,
                                     fit_intercept=fit_intercept)

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

        coef, intercept, opt_data = \
            self.solve_glm(X=self.X,
                           y=self.y,
                           loss_func=self.loss_func,
                           loss_kws=self.loss_kws,
                           fit_intercept=self.fit_intercept,
                           lasso_pen=1,
                           lasso_weights=L1_weights,
                           coef_init=opt_init,
                           intercept_init=opt_init_upv,
                           **self.solver_kws)

        return coef, intercept, opt_data

    def loss(self, value, upv=None):
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
