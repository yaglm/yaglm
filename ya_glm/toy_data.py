import numpy as np
from sklearn.utils import check_random_state
from scipy.special import expit
from numbers import Number
from itertools import product


def sample_sparse_lin_reg(n_samples=100, n_features=10, n_responses=1,
                          n_nonzero=5,
                          X_dist='indep',
                          x_corr=0.1,
                          noise_std=1,
                          intercept=0,
                          random_state=None):
    """
    Samples linear regression data with a sparse regression coefficient.
    
    Parameters
    ----------
    n_samples: int
        Number of samples to draw.
    
    n_features: int
        Number of features.

    n_reponse: int
        Number of responses. If n_responses >= 2 then the coefficient matrix is row sparse.
        
    n_nonzero: int
        Number of non-zero features
        
    X_dist: str
        How to sample the X data. Must be one of ['indep', 'corr'].
        X data is always follows a multivariate Gaussian.
        If 'corr', then cov = (1 - corr) * I + corr * 11^T.
        
    x_corr: float
        How correlated the x data are.
        
    noise_std: float
        Noise level.
        
    intercept: float or array-like, shape (n_responses, )
        The true intercept.
        
    random_state: None, int
        The seed.
        
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data.
        
    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The response.
        
    coef_true: array-like, shape (n_features, ) or (n_features, n_responses)
        The true regression coefficient.
        
    intercept_true: float or array-like shape (n_responses, )
        The true intercept.
    """
    rng = check_random_state(random_state)

    # set coefficient
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero,
                           n_responses=n_responses)

    # sample design matrix
    X = sample_X(n_samples=n_samples,
                 n_features=n_features,
                 X_dist=X_dist, x_corr=x_corr,
                 random_state=rng)

    # set y
    if n_responses == 1:
        E = rng.normal(size=n_samples)

    else:
        E = rng.normal(size=(n_samples, n_responses))
        if isinstance(intercept, Number):
            intercept = intercept * np.ones(n_responses)

    y = X @ coef + intercept + noise_std * E
    
    return X, y, coef, intercept


def infuse_outliers(y, prop_bad=.1, random_state=None):
    """
    Infuses outliers sampled from a T distribution into a response vector.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The original response vector.

    prop_bad: float
        Proportion of y values that should come from outlier distribution.

    random_state: None, int
        See for sampling outliers.

    Output
    ------
    y_bad: array-like, shape (n_samples, )
        The response vector with outliers.
    """
    # TODO: add multivariate
    rng = check_random_state(random_state)

    n_outliers = int(len(y) * prop_bad)

    if y.ndim == 1:
        bad_idxs = rng.choice(a=np.arange(len(y)), size=n_outliers,
                              replace=False)
    else:
        raise NotImplementedError
    # else:
    #     # TODO: is there a more straightforward way of doing this?
    #     bad_meta_idxs = rng.choice(a=np.arange(y.shape[0] * y.shape[1]),
    #                                size=n_outliers, replace=False)
    #     all_idxs = list(product(np.arange(y.shape[0]), np.arange(y.shape[1])))
    #     idxs = all_idxs[bad_meta_idxs]

    bad_y_vals = rng.standard_t(df=1, size=n_outliers)

    y_bad = y.copy()
    y_bad[bad_idxs] = bad_y_vals

    return y_bad


def sample_sparse_log_reg(n_samples=100, n_features=10, n_nonzero=5,
                          X_dist='indep',
                          x_corr=0.1,
                          coef_scale=1,
                          intercept=0,
                          random_state=None):
    """
    Samples logistic regression data with a sparse regression coefficient.

    Parameters
    ----------
    n_samples: int
        Number of samples to draw.

    n_features: int
        Number of features.

    n_nonzero: int
        Number of non-zero features

    X_dist: str
        How to sample the X data. Must be one of ['indep', 'corr'].
        X data is always follows a multivariate Gaussian.
        If 'corr', then cov = (1 - corr) * I + corr * 11^T.

    x_corr: float
        How correlated the x data are.

    coef_scale: float
        The scale of the non-zero entries of the coefficeint.

    intercept: float
        The true intercept.

    random_state: None, int
        The seed.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data.

    y: array-like, shape (n_samples, )
        The binary response

    p: array-like, shape (n_samples, )
        The true probabilities.

    coef_true: array-like, shape (n_features, )
        The true regression coefficient.

    intercept_true: float
        The true intercept.
    """
    rng = check_random_state(random_state)

    # set coefficient
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero)
    coef *= coef_scale

    # sample design matrix
    X = sample_X(n_samples=n_samples,
                 n_features=n_features,
                 X_dist=X_dist, x_corr=x_corr,
                 random_state=rng)

    z = X @ coef + intercept
    p = expit(z)
    y = rng.binomial(n=1, p=p)

    return X, y, p, coef, intercept


def get_sparse_coef(n_features=10, n_nonzero=5, n_responses=1):
    """
    Sets a sparse coefficeint vector where half the entries are positive
    and half the entries are negative.

    Parameters
    ----------
    n_features: int
        Number of features.

    n_nonzero: int
        Number of non-zero features.
    """
    assert n_nonzero <= n_features
    # setup true coefficient
    n_pos = n_nonzero // 2

    if n_responses == 1:
        coef = np.zeros(n_features)
        coef[0:n_pos] = 1
        coef[n_pos:n_nonzero] = -1
        return coef

    else:
        coef = np.zeros((n_features, n_responses))
        coef[0:n_pos, :] = 1
        coef[n_pos:n_nonzero, :] = -1
        return coef


def sample_X(n_samples=100, n_features=10, X_dist='indep', x_corr=0.1,
             random_state=None):
    """
    Samples a random design matrix.

    Parameters
    ----------
    n_samples: int
        Number of samples to draw.

    n_features: int
        Number of features.

    X_dist: str
        How to sample the X data. Must be one of ['indep', 'corr'].
        X data is always follows a multivariate Gaussian.
        If 'corr', then cov = (1 - corr) * I + corr * 11^T.

    x_corr: float
        How correlated the x data are.

    random_state: None, int
        The seed.

    Output
    ------
    X: array-like, (n_samples, n_features)

    """
    rng = check_random_state(random_state)

    assert X_dist in ['indep', 'corr']

    # sample X data
    if X_dist == 'indep':
        return rng.normal(size=(n_samples, n_features))

    elif X_dist == 'corr':
        cov = (1 - x_corr) * np.eye(n_features) + \
            x_corr * np.ones((n_features, n_features))

        return rng.multivariate_normal(mean=np.zeros(n_features), cov=cov,
                                       size=n_samples)
