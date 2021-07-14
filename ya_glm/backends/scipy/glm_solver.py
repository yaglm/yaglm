from ya_glm.backends.scipy.quantile_lin_prog import solve_lin_prog
from ya_glm.backends.scipy.quantile_quad import solve_quad_prog


def solve_glm(X, y,
              loss_func='quantile',
              loss_kws={'quantile': 0.5},
              fit_intercept=True,

              lasso_pen=None,
              lasso_weights=None,

              ridge_pen=None,
              ridge_weights=None,
              tikhonov=None,

              coef_init=None,
              intercept_init=None,

              solver='default',
              tol=None,
              options=None
              ):

    quantile = loss_kws['quantile']

    if lasso_weights is not None and lasso_pen is None:
        lasso_pen = 1

    if (ridge_weights is not None or tikhonov is not None) \
            and ridge_pen is None:
        ridge_pen = 1

    kws = {'X': X,
           'y': y,
           'fit_intercept': fit_intercept,
           'quantile': quantile,
           'lasso_pen': lasso_pen,
           'lasso_weights': lasso_weights,
           'sample_weights': None,  # TODO: add
           'tol': tol}

    if ridge_pen is None:
        if solver == 'default':
            solver = 'highs'

        return solve_lin_prog(solver=solver,
                              **kws)

    else:
        if solver == 'default':
            solver = None

        return solve_quad_prog(ridge_pen=ridge_pen,
                               ridge_weights=ridge_weights,
                               tikhonov=tikhonov,
                               coef_init=coef_init,
                               intercept_init=intercept_init,
                               solver=solver,
                               options=options,
                               **kws)
