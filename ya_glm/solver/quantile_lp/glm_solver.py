from ya_glm.solver.quantile_lp.scipy_lin_prog import solve as solve_lin_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve as solve_quad_prog
from ya_glm.solver.quantile_lp.cvxpy_quad_prog import solve_path

# from warnings import warn


def solve_glm(X, y,
              loss_func='quantile',
              loss_kws={},
              fit_intercept=True,
              sample_weight=None,

              lasso_pen_val=None,
              lasso_weights=None,

              ridge_pen_val=None,
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

    # TODO: perhaps uncomment this warning?
    # but it gets really annoying for the LLA algorithm
    # if coef_init is not None or intercept_init is not None:
    #     warn("Initialization does not currently work for this solver")

    if lasso_weights is not None and lasso_pen_val is None:
        lasso_pen_val = 1

    if (ridge_weights is not None or tikhonov is not None) \
            and ridge_pen_val is None:
        ridge_pen_val = 1

    quantile = loss_kws['quantile']

    kws = {'X': X,
           'y': y,
           'fit_intercept': fit_intercept,
           'quantile': quantile,
           'lasso_pen_val': lasso_pen_val,
           'lasso_weights': lasso_weights,
           'sample_weight': sample_weight,  # TODO: add
           **solver_kws}

    if ridge_pen_val is None:
        if solver == 'default':
            solver = 'highs'

        return solve_lin_prog(solver=solver,
                              **kws)

    else:
        if solver == 'default':
            solver = 'ECOS'

        return solve_quad_prog(**kws,
                               ridge_pen_val=ridge_pen_val,
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
