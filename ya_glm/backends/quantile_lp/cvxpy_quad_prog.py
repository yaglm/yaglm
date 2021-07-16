import cvxpy as cp
from time import time

from ya_glm.backends.fista.glm_solver import process_param_path
from ya_glm.backends.quantile_lp.utils import get_lin_prog_data, \
    get_coef_inter, get_quad_mat


def solve(X, y, fit_intercept=True, quantile=0.5, sample_weight=None,
          lasso_pen=1, ridge_pen=None,
          lasso_weights=None, ridge_weights=None, tikhonov=None,
          coef_init=None, intercept_init=None,
          solver=None,
          cp_kws={}):
    """
    Solves the L1 + L2 penalized quantile regression problem by formulating it as a linear quadratic program then appealing to cvxpy.
    """

    if lasso_weights is not None and lasso_pen is None:
        lasso_pen = 1

    if (ridge_weights is not None or tikhonov is not None) \
            and ridge_pen is None:
        ridge_pen = 1

    start_time = time()

    problem, var, lasso_pen, ridge_pen = \
        setup_problem(X=X, y=y,
                      fit_intercept=fit_intercept,
                      quantile=quantile,
                      sample_weight=sample_weight,
                      lasso_pen=lasso_pen,
                      ridge_pen=ridge_pen,
                      lasso_weights=lasso_weights,
                      ridge_weights=ridge_weights,
                      tikhonov=tikhonov,
                      coef_init=coef_init,
                      intercept_init=intercept_init)

    problem.solve(solver=solver, **cp_kws)
    # solve_with_backups(problem=problem, variable=var, **cp_kws)

    opt_data = {**problem.solver_stats.__dict__,
                'status': problem.status,
                'runtime': time() - start_time}
        
    if fit_intercept:
        n_params = X.shape[1] + 1
    else:
        n_params = X.shape[1]

    coef, intercept = get_coef_inter(solution=var.value,
                                     n_params=n_params,
                                     fit_intercept=fit_intercept)

#     coef = clip_zero(coef, zero_tol=zero_tol)
#     if fit_intercept:
#         intercept = clip_zero(intercept, zero_tol=zero_tol)
#     else:
#         intercept = None

    return coef, intercept, opt_data


def solve_path(fit_intercept=True, cp_kws={}, zero_tol=1e-8,
               lasso_pen_seq=None, ridge_pen_seq=None,
               check_decr=True, **kws):

    param_path = process_param_path(lasso_pen_seq=lasso_pen_seq,
                                    ridge_pen_seq=ridge_pen_seq,
                                    check_decr=check_decr)

    # make sure we setup the right penalty
    if 'lasso_pen' in param_path[0]:
        kws['lasso_pen'] = param_path[0]['lasso_pen']
    if 'ridge_pen' in param_path[0]:
        kws['ridge_pen'] = param_path[0]['ridge_pen']

    start_time = time()
    problem, var, lasso_pen, ridge_pen = setup_problem(**kws)
    pre_setup_runtime = time() - start_time

    for params in param_path:
        start_time = time()

        if 'lasso_pen' in params:
            lasso_pen.value = params['lasso_pen']

        if 'ridge_pen' in params:
            ridge_pen.value = params['ridge_pen']

        problem.solve(**cp_kws)
        # solve_with_backups(problem=problem, variable=var, **cp_kws)

        if var.value is None:
            raise RuntimeError("cvxpy solvers failed")

        opt_data = {**problem.solver_stats.__dict__,
                    'status': problem.status,
                    'runtime': time() - start_time,
                    'pre_setup_runtime': pre_setup_runtime}

        if fit_intercept:
            n_params = kws['X'].shape[1] + 1
        else:
            n_params = kws['X'].shape[1]

        coef, intercept = get_coef_inter(solution=var.value,
                                         n_params=n_params,
                                         fit_intercept=fit_intercept)

    #     coef = clip_zero(coef, zero_tol=zero_tol)
    #     if fit_intercept:
    #         intercept = clip_zero(intercept, zero_tol=zero_tol)
    #     else:
    #         intercept = None

        fit_out = {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}
        yield fit_out, params


def setup_problem(X, y, fit_intercept=True, quantile=0.5, sample_weight=None,
                  lasso_pen=1, ridge_pen=None,
                  lasso_weights=None, ridge_weights=None, tikhonov=None,
                  coef_init=None, intercept_init=None):

    if lasso_pen is not None:
        lasso_pen = cp.Parameter(pos=True, value=lasso_pen)

    if ridge_pen is not None:
        ridge_pen = cp.Parameter(pos=True, value=ridge_pen)

    if coef_init is not None or intercept_init is not None:
        raise NotImplementedError("I do not think initialization works for these solvers")

    ######################
    # setup problem data #
    ######################
    A_eq, b_eq, lin_coef, n_params = \
        get_lin_prog_data(X, y,
                          fit_intercept=fit_intercept,
                          quantile=quantile,
                          lasso_pen=lasso_pen,
                          sample_weight=sample_weight,
                          lasso_weights=lasso_weights)

    lin_coef = cp.hstack(lin_coef)

    if ridge_pen is not None:
        quad_mat = get_quad_mat(X=X,
                                fit_intercept=fit_intercept,
                                weights=ridge_weights,
                                tikhonov=tikhonov)

    n_dim = A_eq.shape[1]
    var = cp.Variable(shape=n_dim)

    ####################
    # setup cp problem #
    ####################
    if ridge_pen is None:
        objective = cp.Minimize(var.T @ lin_coef)
    else:
        objective = cp.Minimize(var.T @ lin_coef +
                                0.5 * ridge_pen * cp.quad_form(var, quad_mat))

    constraints = [var >= 0,
                   A_eq @ var == b_eq]

    problem = cp.Problem(objective, constraints)

    return problem, var, lasso_pen, ridge_pen
