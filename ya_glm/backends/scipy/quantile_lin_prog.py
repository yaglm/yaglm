
import warnings
from time import time

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import OptimizeWarning
from scipy.linalg import LinAlgWarning


def solve_lin_prog(X, y, fit_intercept=True, quantile=0.5, lasso_pen=1,
                   sample_weights=None,
                   lasso_weights=None,
                   tol=None,
                   solver='highs'):
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

    lasso_pen: float
        The multiplicated penalty strength parameter.

    sample_weights: None, array-like shape (n_features, )
        Sample weights

    lasso_weights: None, array-like shape (n_features, )
        Feature weights for the L1 norm.

    tol: None, float
        Tolerance for stopping criteria.

    solver: str
        Which linprog solver to use, see scipy.optimize.linprog

    Output
    ------
    coef, intercept, opt_out
    """
    start_time = time()

    A_eq, b_eq, c, n_params = \
        get_lin_prog_data(X=X, y=y,
                          fit_intercept=fit_intercept,
                          quantile=quantile,
                          lasso_pen=lasso_pen,
                          sample_weights=sample_weights,
                          lasso_weights=lasso_weights)

    if 'highs' in solver:
        options = {'primal_feasibility_tolerance': tol}
    else:
        options = {'tol': tol}

    warnings.filterwarnings('ignore', category=OptimizeWarning)
    warnings.filterwarnings('ignore', category=LinAlgWarning)

    result = linprog(
        c=c,
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


def get_coef_inter(solution, n_params, fit_intercept):
    # positive slack - negative slack
    # solution is an array with (params_pos, params_neg, u, v)
    params = solution[:n_params] - solution[n_params:2 * n_params]

    if fit_intercept:
        coef = params[1:]
        intercept = params[0]
    else:
        coef = params
        intercept = None

    return coef, intercept


def scipy_result_to_dict(result):
    return {'opt_val': result.fun,
            'success': result.success,
            'status': result.status,
            'nit': result.nit,
            'message': result.message}


def get_lin_prog_data(X, y, fit_intercept=True, quantile=0.5, lasso_pen=1,
                      sample_weights=None,
                      lasso_weights=None):
    """

    Output
    ------
    A_eq, b_eq, c, n_params
    """

    n_samples, n_features = X.shape

    # TODO: perhaps filter zero sample weights as in https://github.com/scikit-learn/scikit-learn/blob/0d064cfd4eda6dd4f7c8711a4870d2f02fda52fb/sklearn/linear_model/_quantile.py#L195-L209

    # format sample weights vec
    if sample_weights is None:
        sample_weights = np.ones(n_samples) / n_samples
    else:
        sample_weights = np.array(sample_weights).copy() / n_samples

    # format the L1_vec
    if lasso_weights is None:
        L1_vec = np.ones(n_features)

    else:
        assert len(lasso_weights) == n_features
        L1_vec = np.array(lasso_weights)

    if fit_intercept:
        n_params = n_features + 1
        L1_vec = np.concatenate([[0], L1_vec,  # 0 = do not penalize intercept
                                 [0], L1_vec])
    else:
        n_params = n_features
        L1_vec = np.concatenate([L1_vec, L1_vec])

    # the linear programming formulation of quantile regression
    # follows https://stats.stackexchange.com/questions/384909/

    c = np.concatenate([
        L1_vec * lasso_pen,
        sample_weights * quantile,
        sample_weights * (1 - quantile),
    ])

    if fit_intercept:

        A_eq = np.concatenate([
            np.ones((n_samples, 1)),
            X,
            -np.ones((n_samples, 1)),
            -X,
            np.eye(n_samples),
            -np.eye(n_samples),
        ], axis=1)

    else:
        A_eq = np.concatenate([
            X,
            -X,
            np.eye(n_samples),
            -np.eye(n_samples),
        ], axis=1)

    return A_eq, y, c, n_params
