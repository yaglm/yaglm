import cvxpy as cp
from time import time
from functools import partial

from ya_glm.cvxpy.penalty import lasso, ridge
from ya_glm.cvxpy.loss_functions import lin_reg_loss, log_reg_loss,\
    quantile_reg_loss

from ya_glm.cvxpy.utils import solve_with_backups


def solve_glm(X, y,
              loss_func='lin_reg',
              loss_kws={},
              fit_intercept=True,
              lasso_pen=None,
              lasso_weights=None,
              ridge_pen=None,
              ridge_weights=None,
              coef_init=None,
              intercept_init=None,
              cp_kws={}):

    start_time = time()
    ######################
    # objective function #
    ######################
    glm_loss = get_glm_loss(loss_func)

    if lasso_pen is None and lasso_weights is not None:
        lasso_pen = 1

    if ridge_pen is None and ridge_weights is not None:
        ridge_pen = 1

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

    ###################
    # setup variables #
    ###################

    coef = cp.Variable(shape=X.shape[1], value=coef_init)
    if fit_intercept:
        intercept = cp.Variable(value=intercept_init)
    else:
        intercept = None

    ###########################
    # setup and solve problem #
    ###########################
    problem = cp.Problem(cp.Minimize(objective(coef, intercept)))

    solve_with_backups(problem=problem, variable=coef, **cp_kws)

    if coef.value is None:
        raise RuntimeError("cvxpy solvers failed")

    opt_data = {'runtime': time() - start_time,
                'problem': problem}

    if fit_intercept:
        return coef.value, intercept.value, opt_data

    else:
        return coef.value, None, opt_data


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
