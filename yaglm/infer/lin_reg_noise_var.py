import numpy as np

from yaglm.config.base import Config
from yaglm.autoassign import autoassign
from yaglm.utils import count_support, fit_if_unfitted


class LinRegNoiseEst(Config):
    """
    Base class for estimating the linear regression noise standard deviatio.

    Attributes
    ----------
    scale_: float
        The estimated noise standard deviation.
    """
    pass


class ViaRidge(LinRegNoiseEst):
    """
    Estimates the linear regression noise scale using ridge regression method of (Liu et al, 2020). This estimator only requires fitting a single ridge regression i.e. no tuning.

    Parameters
    ----------
    pen_val: str, float
        The ridge regression penalty parameter to use. If 'default', will use the default choice discussed in (Liu et al, 2020).

    fit_intercept: bool
        Whether or not to fit a ridge regression intercept.

    Attributes
    ----------
    pen_val_: float
        The ridge penalty value we used.

    scale_: float
        The estimated noise standard deviation.

    References
    ----------

    Liu, X., Zheng, S. and Feng, X., 2020. Estimation of error variance via ridge regression. Biometrika, 107(2), pp.481-488.
    """
    @autoassign
    def __init__(self, pen_val='default', fit_intercept=True): pass

    def fit(self, X, y, sample_weight=None):

        sigma_sq, self.pen_val_ = \
            lin_reg_var_via_ridge(X=X, y=y,
                                  fit_intercept=self.fit_intercept,
                                  pen_val=self.pen_val,
                                  sample_weight=sample_weight)
        self.scale_ = np.sqrt(sigma_sq)

        return self


class ViaSelRSS(LinRegNoiseEst):
    """
    Estimates the linear regression noise scale using the residual sum of squares from a fitted estimator (e.g. a Lasso tuned via CV) as discussed in (Reid et al, 2016).

    Parameters
    ----------
    est: Estimator
        Either a fit or unfit estimator e.g. GlmCV(penalty=Lasso()).

    zero_tol: float
        Tolerance for declaring small values equal to zero.

    Attributes
    ----------
    fit_est_: Estimator()
        The fitted estimator.

    scale_: float
        The estimated noise standard deviation.

    References
    ----------
    Reid, S., Tibshirani, R. and Friedman, J., 2016. A study of error variance estimation in lasso regression. Statistica Sinica, pp.35-67.
    """

    @autoassign
    def __init__(self, est, zero_tol=1e-6): pass

    def fit(self, X, y, sample_weight=None):

        # fit an estimator
        self.fit_est_ = fit_if_unfitted(estimator=self.est,
                                        X=X, y=y,
                                        sample_weight=sample_weight)

        # get noise variance estimate
        sigma_sq, self.n_nonzero_ = \
            lin_reg_var_from_rss_of_sel(X, y, coef=self.fit_est_.coef_,
                                        intercept=self.fit_est_.intercept_,
                                        zero_tol=self.zero_tol)

        self.scale_ = np.sqrt(sigma_sq)

        return self


# TODO: add this
class NaturalLasso(LinRegNoiseEst):
    """
    Estimate the linear regression noise standard deviation using the natural Lasso method of (Yu and Bien, 2019).

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """
    def __init__(self):
        raise NotImplementedError("TODO add this")


# TODO: add this
class OrganicLasso(LinRegNoiseEst):
    """
    Estimate the linear regression noise standard deviation using the organic Lasso method of (Yu and Bien, 2019).

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """
    def __init__(self):
        raise NotImplementedError("TODO add this")


# TODO: rewrite formulas so pen_val is on scale where we average the loss
def lin_reg_var_via_ridge(X, y, fit_intercept=True, pen_val='default',
                          sample_weight=None):
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

    sample_weight: None or array-like,  shape (n_samples,)
        (Optional) Individual weights for each sample.

    Output
    ------
    sigma_sq: float
        An estimate of the noise variance.

    pen_val: float
        The penalty value used.

    References
    ----------
    Liu, X., Zheng, S. and Feng, X., 2020. Estimation of error variance via ridge regression. Biometrika, 107(2), pp.481-488.
    """

    if sample_weight is not None:
        raise NotImplementedError

    # if we fit an intercept just assume the intercept is given by the mean
    if fit_intercept:
        y = y - np.array(y).mean()

    n, d = X.shape

    if pen_val == 'default':
        alpha = 0.1
        pen_val = alpha * abs(X.T @ y).max() / (n * d)

    # compute (X^TX + pen_val I)^{-1}
    if n <= d:
        # directly
        xtx_i_inv = np.linalg.inv(X.T @ X / n + pen_val * np.eye(d))
    else:
        # Woodbury matrix identity
        xxt_i_inv = np.linalg.inv(X @ X.T / n + pen_val * np.eye(n))
        xtx_i_inv = (1/pen_val) * (np.eye(d) - (1 / n) * X.T @ xxt_i_inv @ X)

    # see (2) in (Liu et al, 2020)
    A = (1 / n) * X @ xtx_i_inv @ X.T
    sigma_sq_cup = (1 / n) * y.T @ (np.eye(n) - A) @ y
    sigma_sq = sigma_sq_cup / (1 - np.trace(A) / n)

    # other information to return
    # info = {'pen_val': pen_val}

    return sigma_sq, pen_val


# TODO: add sample weight
def lin_reg_var_from_rss_of_sel(X, y, coef, intercept=None, zero_tol=1e-6,
                                sample_weight=None):
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

    sample_weight: None or array-like,  shape (n_samples,)
        (Optional) Individual weights for each sample.

    Output
    ------
    sigma_sq: float
        An estimate of the noise variance.

    n_nonzero: int
        The number of nonzero elements of the estimate: int

    References
    ----------
    Reid, S., Tibshirani, R. and Friedman, J., 2016. A study of error variance estimation in lasso regression. Statistica Sinica, pp.35-67.

    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """
    if sample_weight is not None:
        raise NotImplementedError

    # number of non-zero coefs
    n_nonzero = count_support(coef, zero_tol=zero_tol)

    # TODO: I think add 1 to n_nozero if we have an intercept

    # residual sum of squares
    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    RSS = ((y - y_hat) ** 2).sum()
    sigma_sq = RSS / (X.shape[0] - n_nonzero)

    return sigma_sq, n_nonzero


def lin_reg_var_natural_lasso(X, y, coef, intercept=None, sample_weight=None):
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

    sample_weight: None or array-like,  shape (n_samples,)
        (Optional) Individual weights for each sample.

    Output
    ------
    sigma_sq: float
        An estimate of the noise variance.

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """

    if sample_weight is not None:
        raise NotImplementedError

    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    # See Proposition 1
    sigma_sq = (1 / X.shape[0]) * ((y ** 2).sum() - (y_hat ** 2).sum())

    # See equation (7)
    # RSS = ((y - y_hat) ** 2).sum()
    # sigma_sq = (1 / X.shape[0]) * RSS + 2 * pen_val * abs(coef).sum()

    return sigma_sq


def lin_reg_var_organinc_lasso(X, y, pen_val, coef, intercept=None,
                               sample_weight=None):
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

    sample_weight: None or array-like,  shape (n_samples,)
        (Optional) Individual weights for each sample.

    Output
    ------
    sigma_sq: float
        An estimate of the noise variance.

    References
    ----------
    Yu, G. and Bien, J., 2019. Estimating the error variance in a high-dimensional linear model. Biometrika, 106(3), pp.533-546.
    """

    if sample_weight is not None:
        raise NotImplementedError

    # See equation (17)
    y_hat = X @ coef
    if intercept is not None:
        y_hat += intercept

    RSS = ((y - y_hat) ** 2).sum()
    L1_sq = abs(coef).sum() ** 2
    sigma_sq = (1 / X.shape[0]) * RSS + 2 * pen_val * L1_sq

    return sigma_sq
