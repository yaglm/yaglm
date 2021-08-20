import numpy as np
# TODO: add sample weight to all functions in this module

# TODO: rewrite formulas so pen_val is on scale where we average the loss
def lin_reg_var_via_ridge(X, y, fit_intercept=True, pen_val='default'):
    """
    Estimates the linear regression noise variance use the ridge regression based method from (Liu et al, 2020).

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    fit_intercept: bool
        Whether or not we should include an intercept.

    pen_val: str, float
        The ridge penalty value to use. If 'default', will use the value from section 3 of (Liu et al, 2020)

    Output
    ------
    sigma_sq: float
        An estimate of the noise variance.

    References
    ----------
    Liu, X., Zheng, S. and Feng, X., 2020. Estimation of error variance via ridge regression. Biometrika, 107(2), pp.481-488.
    """

    # if we fit an intercept just assume the intercept is given by the mean
    if fit_intercept:
        y = y - np.array(y).mean()

    n, d = X.shape

    if pen_val == 'default':
        alpha = 0.1
        pen_val = alpha * (X.T @ y) / (n * d)

    # compute (X^TX + pen_val I)^{-1}
    if n <= d:
        # directly
        xtx_i_inv = np.linalg.inv(X.T @ X / n + pen_val)
    else:
        # Woodbury matrix identity
        xxt_i_inv = np.linalg.inv(X @ X.T / n + pen_val * np.eye(n))
        xtx_i_inv = (1/pen_val) * (np.eye(d) - (1 / n) * X.T @ xxt_i_inv @ Xs)

    # see (2) in (Liu et al, 2020)
    A = (1 / n) * X @ xtx_i_inv @ X.T
    sigma_sq_cup = (1 / n) * y.T @ (np.eye(n) - A) @ y
    sigma_sq = sigma_sq_cup / (1 - np.trace(A) / n)

    return sigma_sq


# TODO: add sample weight
def lin_reg_var_from_rss_of_sel(X, y, coef, intercept=None, zero_tol=1e-5):
    """
    Suppose we have an estimate of the coefficient obtained, for example, by tuning a Lasso penalty via cross validation. A reasonable estimate of the linear regression variance is given by

    sigma_sq = (1 /(n - nnz)) * ||y - X @ coef + intercept||_2^2

    where (coef, intercept) are estimated coefficeint/intercept and nnz is the number of non-zero elements of coef. This is equation (5) of (Yu and Bien, 2019) and was studied in (Reid et al, 2016).

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    coef: array-like, shape (n_features, )
        The estimated coefficient.

    intercept: None, float
        The (optional) estimated intercept.

    zero_tol: float
        Coefficients below this value are counted as zeros (e.g. addresses numerical issues where a coefficient may not be set to exactly zero)


    References
    ----------
    Reid, S., Tibshirani, R. and Friedman, J., 2016. A study of error variance estimation in lasso regression. Statistica Sinica, pp.35-67.

    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """

    # number of non-zero coefs
    n_nonzero = sum(abs(coef) > zero_tol)

    # residual sum of squares
    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    RSS = ((y - y_hat) ** 2).sum()
    sigma_sq = RSS / (X.shape[0] - n_nonzero)

    return sigma_sq


def lin_reg_var_natural_lasso(X, y, coef, intercept=None):
    """
    Estimates the linear regression variance using the natural Lasso estimate (Yu and Bien, 2019). This requires first fitting a Lasso penalized model.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    coef: array-like, shape (n_features, )
        The Lasso estimated coefficient.

    intercept: None, float
        The (optional) estimated intercept.

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """

    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    # See Proposition 1
    sigma_sq = (1 / X.shape[0]) * ((y ** 2).sum() - (y_hat ** 2).sum())

    # See equation (7)
    # RSS = ((y - y_hat) ** 2).sum()
    # sigma_sq = (1 / X.shape[0]) * RSS + 2 * pen_val * abs(coef).sum()

    return sigma_sq


def lin_reg_var_organinc_lasso(X, y, pen_val, coef, intercept=None):
    """
    Estimates the linear regression variance using the organic Lasso estimate (Yu and Bien, 2019). This requires first fitting a L1 squared penalized model.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    pen_val: float
        The squared L1 penalty value.

    coef: array-like, shape (n_features, )
        The Lasso estimated coefficient.

    intercept: None, float
        The (optional) estimated intercept.

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """

    # See equation (17)
    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    RSS = ((y - y_hat) ** 2).sum()
    L1_sq = abs(coef).sum() ** 2
    sigma_sq = (1 / X.shape[0]) * RSS + 2 * pen_val * L1_sq

    return sigma_sq
