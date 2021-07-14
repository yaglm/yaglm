import cvxpy as cp
from functools import partial
from time import time

from ya_glm.utils import clip_zero
from ya_glm.cvxpy.penalty import lasso, ridge
from ya_glm.cvxpy.loss_functions import lin_reg_loss, log_reg_loss,\
    quantile_reg_loss
from ya_glm.cvxpy.utils import solve_with_backups
from ya_glm.backends.fista.glm_solver import process_param_path


def solve_glm(X, y,
              loss_func='lin_reg',
              loss_kws={},
              fit_intercept=True,
              lasso_pen=None,
              lasso_weights=None,
              groups=None,
              L1to2=False,
              nuc=False,
              ridge_pen=None,
              ridge_weights=None,
              tikhonov=None,
              coef_init=None,
              intercept_init=None,
              zero_tol=1e-8,
              cp_kws={}):

    start_time = time()
    ######################
    # objective function #
    ######################

    problem, coef, intercept, _, __ =  \
        setup_problem(X=X, y=y,
                      loss_func=loss_func,
                      loss_kws=loss_kws,
                      fit_intercept=fit_intercept,
                      lasso_pen=lasso_pen,
                      lasso_weights=lasso_weights,
                      groups=groups,
                      L1to2=L1to2,
                      nuc=nuc,
                      ridge_pen=ridge_pen,
                      ridge_weights=ridge_weights,
                      coef_init=coef_init,
                      intercept_init=intercept_init)

    solve_with_backups(problem=problem, variable=coef, **cp_kws)

    if coef.value is None:
        raise RuntimeError("cvxpy solvers failed")

    coef, intercept, opt_data = \
        process_output(problem=problem, coef=coef,
                       intercept=intercept,
                       fit_intercept=fit_intercept,
                       zero_tol=zero_tol)

    opt_data['runtime'] = time() - start_time

    return coef, intercept, opt_data


def solve_glm_path(fit_intercept=True, cp_kws={}, zero_tol=1e-8,
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
    problem, coef, intercept, lasso_pen, ridge_pen = setup_problem(**kws)
    pre_setup_runtime = start_time - time()

    for params in param_path:
        start_time = time()

        if 'lasso_pen' in params:
            lasso_pen.value = params['lasso_pen']

        if 'ridge_pen' in params:
            ridge_pen.value = params['ridge_pen']

        solve_with_backups(problem=problem, variable=coef, **cp_kws)

        if coef.value is None:
            raise RuntimeError("cvxpy solvers failed")

        # format output
        fit_out = process_output(problem=problem, coef=coef,
                                 intercept=intercept,
                                 fit_intercept=fit_intercept,
                                 zero_tol=zero_tol)

        fit_out = {'coef': fit_out[0],
                   'intercept': fit_out[1],
                   'opt_data': fit_out[2]}

        fit_out['opt_data']['runtime'] = start_time - time()
        fit_out['opt_data']['pre_setup_runtime'] = pre_setup_runtime

        yield fit_out, params


def process_output(problem, coef, intercept, fit_intercept, zero_tol):

    opt_data = {**problem.solver_stats.__dict__,
                'status': problem.status}

    coef_sol = clip_zero(coef.value, zero_tol=zero_tol)
    if fit_intercept:
        intercept_sol = clip_zero(intercept.value, zero_tol=zero_tol)
    else:
        intercept_sol = None

    return coef_sol, intercept_sol, opt_data


def setup_problem(X, y,
                  loss_func='lin_reg',
                  loss_kws={},
                  fit_intercept=True,
                  lasso_pen=None,
                  lasso_weights=None,
                  groups=None,
                  L1to2=False,
                  nuc=False,
                  ridge_pen=None,
                  ridge_weights=None,
                  tikhonov=None,
                  coef_init=None,
                  intercept_init=None):
    """

    Output
    ------
    problem, coef, intercept, lasso_pen, ridge_pen

    """
    glm_loss = get_glm_loss(loss_func)

    # TODO: add these
    if tikhonov is not None:
        raise NotImplementedError
    if groups is not None:
        raise NotImplementedError
    if L1to2:
        raise NotImplementedError
    if nuc:
        raise NotImplementedError

    if lasso_pen is None and lasso_weights is not None:
        lasso_pen = 1

    if ridge_pen is None and ridge_weights is not None:
        ridge_pen = 1

    if lasso_pen is not None:
        lasso_pen = cp.Parameter(nonneg=True, value=lasso_pen)

    if ridge_pen is not None:
        ridge_pen = cp.Parameter(nonneg=True, value=ridge_pen)

    if lasso_pen is not None and ridge_pen is not None:

        def objective(coef, intercept):
            return glm_loss(X=X, y=y, coef=coef, intercept=intercept) + \
                 lasso_pen * lasso(coef, weights=lasso_weights) + \
                 ridge_pen * ridge(coef, weights=ridge_weights)

    elif lasso_pen is not None:
        def objective(coef, intercept):
            return glm_loss(X=X, y=y, coef=coef, intercept=intercept) + \
                 lasso_pen * lasso(coef, weights=lasso_weights)

    elif ridge_pen is not None:
        def objective(coef, intercept):
            return glm_loss(X=X, y=y, coef=coef, intercept=intercept) + \
                  ridge_pen * ridge(coef, weights=ridge_weights)

    else:
        def objective(coef, intercept):
            return glm_loss(X=X, y=y, coef=coef, intercept=intercept)

    ###############################
    # setup variables and problem #
    ###############################

    coef = cp.Variable(shape=X.shape[1], value=coef_init)
    if fit_intercept:
        intercept = cp.Variable(value=intercept_init)
    else:
        intercept = None

    problem = cp.Problem(cp.Minimize(objective(coef, intercept)))

    return problem, coef, intercept, lasso_pen, ridge_pen


def get_glm_loss(loss_func, loss_kws={}):

    if loss_func == 'lin_reg':
        loss = lin_reg_loss

    elif loss_func == 'log_reg':
        loss = log_reg_loss

    elif loss_func == 'quantile':
        loss = quantile_reg_loss

    if len(loss_kws) > 0:
        loss = partial(loss, **loss_kws)

    return loss