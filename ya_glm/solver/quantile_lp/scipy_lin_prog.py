import warnings
from time import time

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
from scipy.linalg import LinAlgWarning

from ya_glm.solver.quantile_lp.utils import get_lin_prog_data, get_coef_inter


def solve(X, y, fit_intercept=True, quantile=0.5, lasso_pen_val=1,
          sample_weight=None,
          lasso_weights=None,
          solver='highs',
          tol=None
          ):
    """
    Solves the L1 penalized quantile regression problem using scipy's linprog solver. The code is adapted from https://github.com/benchopt/benchmark_quantile_regression and https://github.com/scikit-learn/scikit-learn/blob/0d064cfd4eda6dd4f7c8711a4870d2f02fda52fb/sklearn/linear_model/_quantile.py#L195-L209

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, )
        The training response data.

    fit_intercept: bool
        Whether or not to fit an intercept.

    quantile: float
        Which quantile.

    lasso_pen_val: float
        The multiplicated penalty strength parameter.

    sample_weight: None, array-like shape (n_features, )
        Sample weights

    lasso_weights: None, array-like shape (n_features, )
        Feature weights for the L1 norm.

    solver: str
        Which linprog solver to use, see scipy.optimize.linprog

    tol: None, float
        Tolerance for stopping criteria.

    Output
    ------
    coef, intercept, opt_out
    """
    start_time = time()

    A_eq, b_eq, c, n_params = \
        get_lin_prog_data(X=X, y=y,
                          fit_intercept=fit_intercept,
                          quantile=quantile,
                          lasso_pen_val=lasso_pen_val,
                          sample_weight=sample_weight,
                          lasso_weights=lasso_weights)

    if 'highs' in solver:
        options = {'primal_feasibility_tolerance': tol}
    else:
        options = {'tol': tol}

    warnings.filterwarnings('ignore', category=OptimizeWarning)
    warnings.filterwarnings('ignore', category=LinAlgWarning)

    result = linprog(
        c=np.concatenate(c),
        A_eq=A_eq,
        b_eq=b_eq,
        method=solver,
        options=options
    )

    coef, intercept = get_coef_inter(solution=result.x,
                                     n_params=n_params,
                                     fit_intercept=fit_intercept)

    opt_out = scipy_result_to_dict(result)
    opt_out['runtime'] = time() - start_time

    return coef, intercept, opt_out


def scipy_result_to_dict(result):
    return {'opt_val': result.fun,
            'success': result.success,
            'status': result.status,
            'nit': result.nit,
            'message': result.message}
