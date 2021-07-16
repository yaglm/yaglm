from ya_glm.lla.WeightedLassoSolver import WL1SolverGlm as _WL1SolverGlm


class WL1SolverGlm(_WL1SolverGlm):

    # this WL1 solver will precompute some data e.g. Lipschitz constants
    # to speed up the solver

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

                           # This is the only difference!!
                           loss_func=self.glm_loss,

                           loss_kws=self.loss_kws,
                           fit_intercept=self.fit_intercept,
                           lasso_pen=1,
                           lasso_weights=L1_weights,
                           coef_init=opt_init,
                           intercept_init=opt_init_upv,
                           **self.solver_kws)

        return coef, intercept, opt_data
