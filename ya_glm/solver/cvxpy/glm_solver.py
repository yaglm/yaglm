import cvxpy as cp
from functools import partial
from time import time

from ya_glm.utils import clip_zero
from ya_glm.solver.cvxpy.penalty import lasso_penalty, ridge_penalty,\
     tikhonov_penalty,  multi_task_lasso_penalty, group_lasso_penalty
from ya_glm.solver.cvxpy.loss_functions import lin_reg_loss, log_reg_loss,\
    quantile_reg_loss
from ya_glm.solver.utils import process_param_path


def solve_glm(X, y, loss,
              fit_intercept=True,
              sample_weight=None,

              lasso_pen_val=None,
              lasso_weights=None,
              groups=None,
              multi_task=False,
              nuc=False,
              ridge_pen_val=None,
              ridge_weights=None,
              tikhonov=None,

              coef_init=None,
              intercept_init=None,

              zero_tol=1e-8,
              solver=None,
              cp_kws={}):
    """
    Solves a penalized GLM problem using cvxpy.

    Parameters
    ---------
    zero_tol: float
        Values of the solution smaller than this are set to exactly zero.

    solver: None, str
        Which cvxpy solver to use. See cvxpy docs.

    cp_kws: dict
        Keyword arguments to the call to problem.solve(). See cvxpy docs.
    """

    start_time = time()
    ######################
    # objective function #
    ######################

    problem, coef, intercept, _, __ =  \
        setup_problem(X=X, y=y, loss=loss,
                      fit_intercept=fit_intercept,
                      sample_weight=sample_weight,
                      lasso_pen_val=lasso_pen_val,
                      lasso_weights=lasso_weights,
                      groups=groups,
                      multi_task=multi_task,
                      nuc=nuc,
                      ridge_pen_val=ridge_pen_val,
                      ridge_weights=ridge_weights,
                      coef_init=coef_init,
                      intercept_init=intercept_init)

    problem.solve(solver=solver, **cp_kws)

    if coef.value is None:
        raise RuntimeError("cvxpy solvers failed")

    coef, intercept, opt_data = \
        process_output(problem=problem, coef=coef,
                       intercept=intercept,
                       fit_intercept=fit_intercept,
                       zero_tol=zero_tol)

    opt_data['runtime'] = time() - start_time

    return coef, intercept, opt_data


def solve_glm_path(fit_intercept=True, solver=None, cp_kws={}, zero_tol=1e-8,
                   lasso_pen_seq=None, ridge_pen_seq=None,
                   check_decr=True, **kws):

    param_path = process_param_path(lasso_pen_seq=lasso_pen_seq,
                                    ridge_pen_seq=ridge_pen_seq,
                                    check_decr=check_decr)

    # make sure we setup the right penalty
    if 'lasso_pen_val' in param_path[0]:
        kws['lasso_pen_val'] = param_path[0]['lasso_pen_val']
    if 'ridge_pen_val' in param_path[0]:
        kws['ridge_pen_val'] = param_path[0]['ridge_pen_val']

    start_time = time()
    problem, coef, intercept, lasso_pen_val, ridge_pen_val = \
        setup_problem(**kws)
    pre_setup_runtime = start_time - time()

    for params in param_path:
        start_time = time()

        if 'lasso_pen_val' in params:
            lasso_pen_val.value = params['lasso_pen_val']

        if 'ridge_pen_val' in params:
            ridge_pen_val.value = params['ridge_pen_val']

        problem.solve(solver=solver, **cp_kws)

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

    if hasattr(intercept_sol, 'ndim') and intercept_sol.ndim == 0:
        intercept_sol = float(intercept_sol)

    return coef_sol, intercept_sol, opt_data


def setup_problem(X, y, loss,
                  fit_intercept=True,
                  sample_weight=None,
                  lasso_pen_val=None,
                  lasso_weights=None,
                  groups=None,
                  multi_task=False,
                  nuc=False,
                  ridge_pen_val=None,
                  ridge_weights=None,
                  tikhonov=None,
                  coef_init=None,
                  intercept_init=None):
    """

    Output
    ------
    problem, coef, intercept, lasso_pen_val, ridge_pen_val

    """
    glm_loss = get_glm_loss(loss)

    if nuc:
        raise NotImplementedError

    if sample_weight is not None:
        raise NotImplementedError("need to add")

    if lasso_pen_val is None and lasso_weights is not None:
        lasso_pen_val = 1

    if ridge_pen_val is None and ridge_weights is not None:
        ridge_pen_val = 1

    if ridge_weights is not None:
        assert tikhonov is None

    ###################
    # Setup variables #
    ###################
    if lasso_pen_val is not None:
        lasso_pen_val = cp.Parameter(nonneg=True, value=lasso_pen_val)

    if ridge_pen_val is not None:
        ridge_pen_val = cp.Parameter(nonneg=True, value=ridge_pen_val)

    coef = cp.Variable(shape=X.shape[1], value=coef_init)
    if fit_intercept:
        intercept = cp.Variable(value=intercept_init)
    else:
        intercept = None

    # set objective
    objective = glm_loss(X=X, y=y, coef=coef, intercept=intercept)

    # Add lasso
    if lasso_pen_val is not None:

        if groups is not None:
            objective += lasso_pen_val * \
                group_lasso_penalty(coef, groups=groups, weights=lasso_weights)

        elif multi_task:
            objective += lasso_pen_val * \
                multi_task_lasso_penalty(coef, weights=lasso_weights)

        else:
            objective += lasso_pen_val * lasso_penalty(coef, weights=lasso_weights)

    # Add ridge
    if ridge_pen_val is not None:
        if tikhonov is not None:
            objective += ridge_pen_val * \
                tikhonov_penalty(coef, tikhonov=tikhonov)

        else:
            objective += ridge_pen_val * ridge_penalty(coef, weights=ridge_weights)

    problem = cp.Problem(cp.Minimize(objective))
    return problem, coef, intercept, lasso_pen_val, ridge_pen_val


def get_glm_loss(loss):

    loss_kws = loss.get_loss_kws()

    if loss.name == 'lin_reg':
        loss = lin_reg_loss

    elif loss.name == 'log_reg':
        loss = log_reg_loss
        loss_kws = {}

    elif loss.name == 'quantile':
        loss = quantile_reg_loss

    if len(loss_kws) > 0:
        loss = partial(loss, **loss_kws)

    return loss
