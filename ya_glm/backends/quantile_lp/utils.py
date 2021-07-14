import numpy as np
from scipy.sparse import diags, block_diag, csr_matrix


def get_lin_prog_data(X, y, fit_intercept=True, quantile=0.5, lasso_pen=1,
                      sample_weights=None,
                      lasso_weights=None):
    """
    Gets the data needed for the linear program.

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

    c = [
        lasso_pen * L1_vec,
        sample_weights * quantile,
        sample_weights * (1 - quantile),
    ]

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


def get_quad_mat(X, fit_intercept=True,
                 weights=None, tikhonov=None):
    """
    Returns the ridge penalty matrix for the quadratic program.

    0.5 * param.T @ mat @ param
    """

    n_samples, n_features = X.shape
    dtype = X.dtype

    if weights is not None:
        assert tikhonov is None

    if tikhonov:
        tik_tik = tikhonov.T @ tikhonov
        params_mat = block_diag([tik_tik, tik_tik])

    else:
        if weights is not None:
            diag_elts = np.array(weights)

        else:
            diag_elts = np.ones(n_features)

        if fit_intercept:
            diag_elts = np.concatenate([[0], diag_elts,  # 0 = do not penalize intercept
                                        [0], diag_elts])
        else:
            weights = np.concatenate([diag_elts, diag_elts])

        params_mat = diags(diag_elts, dtype=dtype)

    zero_padding = csr_matrix((n_samples, n_samples), dtype=dtype)
    return block_diag([params_mat, zero_padding, zero_padding])


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
