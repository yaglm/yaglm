from ya_glm.backends.quantile_lp.scipy_lin_prog import solve as solve_lin_prog
from ya_glm.backends.quantile_lp.cvxpy_quad_prog import solve as solve_quad_prog
from ya_glm.backends.quantile_lp.cvxpy_quad_prog import solve_path


def solve_glm(X, y,
              loss_func='quantile',
              loss_kws={},
              fit_intercept=True,

              lasso_pen=None,
              lasso_weights=None,

              ridge_pen=None,
              ridge_weights=None,
              tikhonov=None,

              coef_init=None,
              intercept_init=None,

              solver='default',
              solver_kws={}
              ):
    """
    Solves quantile regression with either a Linear Programming (LP) formulation (for unpenalizer or Lasso penalties) or a Quadratic Programming (QP) formulation (for ridge penalties). For LPs we uses scipy's linprog solver. For QPs we use cvxpy.



    """
    if loss_func != 'quantile':
        raise NotImplementedError("This solver only works for quantile regression")

    if coef_init is not None or intercept_init is not None:
        raise NotImplementedError("I do not think initialization works for these solvers")

    if lasso_weights is not None and lasso_pen is None:
        lasso_pen = 1

    if (ridge_weights is not None or tikhonov is not None) \
            and ridge_pen is None:
        ridge_pen = 1

    quantile = loss_kws['quantile']

    kws = {'X': X,
           'y': y,
           'fit_intercept': fit_intercept,
           'quantile': quantile,
           'lasso_pen': lasso_pen,
           'lasso_weights': lasso_weights,
           # 'sample_weights': None,  # TODO: add
           **solver_kws}

    if ridge_pen is None:
        if solver == 'default':
            solver = 'highs'

        return solve_lin_prog(solver=solver,
                              **kws)

    else:
        if solver == 'default':
            solver = 'ECOS'

        return solve_quad_prog(**kws,
                               ridge_pen=ridge_pen,
                               ridge_weights=ridge_weights,
                               tikhonov=tikhonov,
                               solver=solver,
                               cp_kws=solver_kws)


def solve_glm_path(loss_func='quantile',
                   loss_kws={}, **kws):
    """
    Path algorithm for the Linear and Quadratic Program formulations of quantile regression solved using cvxpy. This is not a true path algorithm in the sense that (I believe) the solution is not reused. However this does save time by resuing the solver setups.
    """

    if loss_func != 'quantile':
        raise NotImplementedError("This solver only works for quantile regression")

    quantile = loss_kws.pop('quantile', 0.5)
    return solve_path(loss_func=loss_func, quantile=quantile, **kws)
