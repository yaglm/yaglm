# import numpy as np
from ya_glm.lla.WeightedLassoSolver import WeightedLassoSolver
from ya_glm.lla.utils import safe_concat

from ya_glm.backends.fista.glm_solver import get_glm_loss as get_glm_loss_fista
from .glm_solver import solve_glm


class WL1SolverGlm(WeightedLassoSolver):
    def __init__(self,  X, y, loss_func, loss_kws={},
                 fit_intercept=True,
                 opt_kws={}):

        self.glm_loss = get_glm_loss_fista(X=X, y=y,
                                           loss_func=loss_func,
                                           loss_kws=loss_kws,
                                           fit_intercept=fit_intercept,
                                           precomp_lip=None)

        self.loss_func = loss_func
        self.loss_kws = loss_kws

        self.opt_kws = opt_kws

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
            solve_glm(X=self.glm_loss.X,
                      y=self.glm_loss.y,
                      loss_func=self.loss_func,
                      loss_kws=self.loss_kws,
                      fit_intercept=self.glm_loss.fit_intercept,
                      lasso_pen=1,
                      lasso_weights=L1_weights,
                      coef_init=opt_init,
                      intercept_init=opt_init_upv,
                      # groups=self.groups,
                      # L1to2=self.L1to2,
                      # nuc=self.nuc,
                      **self.opt_kws)

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
            #return self.glm_loss.eval(np.concatenate([[upv], value]))
            return self.glm_loss.eval(safe_concat(upv, value))

        else:
            return self.glm_loss.eval(value)
