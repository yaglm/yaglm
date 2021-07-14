from scipy.sparse import diags, block_diag, csr_matrix
from scipy.optimize import minimize, Bounds, LinearConstraint
import numpy as np
from time import time

from ya_glm.backends.scipy.quantile_lin_prog import get_lin_prog_data, \
    get_coef_inter, scipy_result_to_dict


def solve_quad_prog(X, y, fit_intercept=True, quantile=0.5, sample_weights=None,
                    lasso_pen=1, ridge_pen=1,
                    lasso_weights=None, ridge_weights=None, tikhonov=None,
                    coef_init=None, intercept_init=None,
                    solver=None, tol=None, options=None):
    """
    Solves the L1 + L2 penalized quantile regression problem using scipy's minimize function.
    """

    start_time = time()

    ######################
    # setup problem data #
    ######################
    A_eq, b_eq, lin_coef, n_params = \
        get_lin_prog_data(X, y,
                          fit_intercept=fit_intercept,
                          quantile=quantile,
                          lasso_pen=lasso_pen,
                          sample_weights=sample_weights,
                          lasso_weights=lasso_weights)

    quad_mat = get_ridge_mat(X=X,
                             fit_intercept=fit_intercept,
                             pen_val=ridge_pen,
                             weights=ridge_weights,
                             tikhonov=tikhonov)

    n_dim = A_eq.shape[1]

    bounds = Bounds(lb=np.zeros(n_dim),
                    ub=np.array([np.inf] * n_dim))

    lin_constr = LinearConstraint(A=A_eq, lb=b_eq, ub=b_eq)

    fun = func_getter(lin_coef=lin_coef, quad_mat=quad_mat, level=1)

    # setup initializer
    if coef_init is not None:
        # if intercept_init is None:
        #     intercept_init = 0
        raise NotImplementedError("TODO add this")

    else:
        x0 = np.zeros(n_dim)

    #########
    # Solve #
    #########

    result = minimize(fun=fun, x0=x0, jac=True, hess=False,
                      bounds=bounds, constraints=lin_constr,
                      method=solver, tol=tol, options=options)

    coef, intercept = get_coef_inter(solution=result.x,
                                     n_params=n_params,
                                     fit_intercept=fit_intercept)

    opt_out = scipy_result_to_dict(result)
    opt_out['runtime'] = time() - start_time

    return coef, intercept, opt_out


def get_ridge_mat(X, fit_intercept=True,
                  pen_val=None, weights=None, tikhonov=None):
    """
    Returns the ridge penalty matrix part of the loss that looks like
    
    0.5 * param.T @ mat @ param
    """
    
    n_samples, n_features = X.shape
    dtype = X.dtype
    
    if (weights is not None or tikhonov is not None) and pen_val is None:
        pen_val = 1.0
    if weights is not None:
        assert tikhonov is None
    
    if pen_val is None:
        return None
    
    if tikhonov:
        tik_tik = tikhonov.T @ tikhonov
        params_mat = pen_val * block_diag([tik_tik, tik_tik])
    
    else:
        if weights is not None:
            diag_elts = pen_val * np.array(weights)
        
        else:
            diag_elts = pen_val * np.ones(n_features)

        if fit_intercept:
            diag_elts = np.concatenate([[0], diag_elts,  # 0 = do not penalize intercept
                                        [0], diag_elts])
        else:
            weights = np.concatenate([diag_elts, diag_elts])

        params_mat = diags(diag_elts, dtype=dtype)

    zero_padding = csr_matrix((n_samples, n_samples), dtype=dtype)
    return block_diag([params_mat, zero_padding, zero_padding])
        
    
def func_getter(lin_coef, quad_mat, level=2):
    
    if level == 0:
        def func(x):
            return lin_coef.T @ x + 0.5 * x.T @ quad_mat @ x
        
        return func
    
    def func_jac(x):
        q_x = quad_mat @ x
        val = lin_coef.T @ x + 0.5 * x.T @ q_x
        grad = lin_coef + q_x

        return val, grad
    
    if level == 1:
        return func_jac

    else:
        def func_jac_hess(x):
            val, grad = func_jac(x)
            return val, grad, quad_mat
            
        return func_jac_hess
