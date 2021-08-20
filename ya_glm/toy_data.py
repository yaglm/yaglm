import numpy as np
from sklearn.utils import check_random_state
from scipy.special import expit
from sklearn.utils.extmath import softmax
from numbers import Number

# TODO: add signal to noise ratio for logistic, multinomial and poisson


def sample_sparse_lin_reg(n_samples=100, n_features=10,
                          n_responses=1,
                          n_nonzero=5,
                          cov='ar',
                          corr=0.35,
                          beta_type=1,
                          snr=1,
                          noise_std=None,
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

    beta_type: int
        Which type of coefficient to create; see Section 3.1 of (Hastie et al, 2017).

    cov: str
        The covariance matrix of the X data. Must be one of ['ident', 'tot', 'ar']. If cov == 'ident' then we use the identity. If cov == 'ar' then Sigma_{ij} = corr **|i-j| follows an autoregression process. If cov == 'tot' then  (1 - corr) * I + corr * 11^T.

    corr: float
        The correlation for the covariace matrix for the X data.
        
    snr: float
        The desired signal to noise ratio defined as coef.T @ cov @ coef / noise_std ** 2. This will automatically set noise_std.

    noise_std: None, float
        The (optional) user specified noise level; this will over ride the snr argument if provided.
        
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
        
    info: dict
        Information related to the true sample distribution e.g. the true coefficient.
    """
    rng = check_random_state(random_state)

    # set coefficient and covariance matrix
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero,
                           n_responses=n_responses, beta_type=beta_type)

    cov = get_cov(n_features=n_features, cov=cov, corr=corr)

    # determine the noise_std
    ct_Sigma_c = coef_cov_quad_form(coef, cov)
    if noise_std is None:
        noise_std = np.sqrt(ct_Sigma_c / snr)
    else:
        snr = ct_Sigma_c / (noise_std ** 2)

    # sample X data
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov,
                                size=n_samples)

    # sample noise
    if n_responses == 1:
        E = rng.normal(size=n_samples)

    else:
        E = rng.normal(size=(n_samples, n_responses))
        if isinstance(intercept, Number):
            intercept = intercept * np.ones(n_responses)

    # set y
    y = X @ coef + intercept + noise_std * E

    # other information
    info = {'coef': coef, 'intercept': intercept,
            'snr': snr, 'noise_std': noise_std,
            'cov': cov}
    
    return X, y, info


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


# TODO: add signal to noise ratio
def sample_sparse_log_reg(n_samples=100, n_features=10, n_nonzero=5,
                          beta_type=1,
                          cov='ar',
                          corr=0.35,
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

    beta_type: int
        Which type of coefficient to create; see Section 3.1 of (Hastie et al, 2017).

    cov: str
        The covariance matrix of the X data. Must be one of ['ident', 'tot', 'ar']. If cov == 'ident' then we use the identity. If cov == 'ar' then Sigma_{ij} = corr **|i-j| follows an autoregression process. If cov == 'tot' then  (1 - corr) * I + corr * 11^T.

    corr: float
        The correlation for the covariace matrix for the X data.

    coef_scale: float
        The value of coef.T @ cov @ coef, which controls the signal to noise ratio.

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

    info: dict
        Information related to the true sample distribution e.g. the true coefficient.
    """
    rng = check_random_state(random_state)

    # set coefficient and covariance matrix
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero,
                           beta_type=beta_type)

    cov = get_cov(n_features=n_features, cov=cov, corr=corr)

    # determine the noise_std
    ct_Sigma_c = coef_cov_quad_form(coef, cov)
    if ct_Sigma_c > np.finfo(float).eps:
        coef *= (coef_scale / ct_Sigma_c)

    # sample X data
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov,
                                size=n_samples)

    z = X @ coef + intercept
    p = expit(z)
    y = rng.binomial(n=1, p=p)

    # other information
    info = {'coef': coef, 'intercept': intercept,
            'coef_scale': coef_scale,
            'probs': p}

    return X, y, info


def sample_sparse_multinomial(n_samples=100, n_features=10,
                              n_nonzero=5, n_classes=3, beta_type=1,
                              cov='ar',
                              corr=0.35,
                              coef_scale=1,
                              intercept=0,
                              random_state=None):
    """
    Samples multinomial regression data with a sparse regression coefficient.

    Parameters
    ----------
    n_samples: int
        Number of samples to draw.

    n_features: int
        Number of features.

    n_nonzero: int
        Number of non-zero features

    beta_type: int
        Which type of coefficient to create; see Section 3.1 of (Hastie et al, 2017).

    cov: str
        The covariance matrix of the X data. Must be one of ['ident', 'tot', 'ar']. If cov == 'ident' then we use the identity. If cov == 'ar' then Sigma_{ij} = corr **|i-j| follows an autoregression process. If cov == 'tot' then  (1 - corr) * I + corr * 11^T.

    corr: float
        The correlation for the covariace matrix for the X data.

    coef_scale: float
        The value of coef.T @ cov @ coef, which controls the signal to noise ratio.

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

    info: dict
        Information related to the true sample distribution e.g. the true coefficient.
    """
    rng = check_random_state(random_state)

    # set coefficient and covariance matrix
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero,
                           beta_type=beta_type, n_responses=n_classes)

    cov = get_cov(n_features=n_features, cov=cov, corr=corr)

    # determine the noise_std
    ct_Sigma_c = coef_cov_quad_form(coef, cov)
    if ct_Sigma_c > np.finfo(float).eps:
        coef *= (coef_scale / ct_Sigma_c)

    # sample X data
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov,
                                size=n_samples)
    z = X @ coef + intercept
    p = softmax(z)

    classes = np.arange(n_classes)
    y = np.array([np.random.choice(a=classes,  p=p[i, :])
                  for i in range(n_samples)])

    # other information
    info = {'coef': coef, 'intercept': intercept,
            'coef_scale': coef_scale,
            'probs': p}

    return X, y, info


def sample_sparse_poisson_reg(n_samples=100, n_features=10, n_responses=1,
                              n_nonzero=5,
                              beta_type=1,
                              coef_scale=1,
                              cov='ar',
                              corr=0.35,
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

    beta_type: int
        Which type of coefficient to create; see Section 3.1 of (Hastie et al, 2017).

    cov: str
        The covariance matrix of the X data. Must be one of ['ident', 'tot', 'ar']. If cov == 'ident' then we use the identity. If cov == 'ar' then Sigma_{ij} = corr **|i-j| follows an autoregression process. If cov == 'tot' then  (1 - corr) * I + corr * 11^T.

    corr: float
        The correlation for the covariace matrix for the X data.

    coef_scale: float
        The value of coef.T @ cov @ coef, which controls the signal to noise ratio.

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

    info: dict
        Information related to the true sample distribution e.g. the true coefficient.
    """
    rng = check_random_state(random_state)

    # set coefficient and covariance matrix
    coef = get_sparse_coef(n_features=n_features, n_nonzero=n_nonzero,
                           beta_type=beta_type)

    cov = get_cov(n_features=n_features, cov=cov, corr=corr)

    # determine the noise_std
    ct_Sigma_c = coef_cov_quad_form(coef, cov)
    if ct_Sigma_c > np.finfo(float).eps:
        coef *= (coef_scale / ct_Sigma_c)

    # sample X data
    X = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov,
                                size=n_samples)
    # set y
    z = X @ coef + intercept
    lam = np.exp(z)
    y = rng.poisson(lam=lam, size=z.shape)

    # other information
    info = {'coef': coef, 'intercept': intercept,
            'coef_scale': coef_scale,
            'lam': lam}

    return X, y, info


def get_sparse_coef(n_features=10, n_nonzero=5, n_responses=1, beta_type=1,
                    neg_idx=None, laplace=False, random_state=None):
    """
    Sets a sparse coefficeint vector where half the entries are positive
    and half the entries are negative.

    Parameters
    ----------
    n_features: int
        Number of features.

    n_nonzero: int
        Number of non-zero features.

    n_responses: int
        Number of responses. For multiple responses, the first coefficient is positive and the remaining responses have a single negative entry.

    beta_type: int
        Which type of coefficient to create; see Section 3.1 of (Hastie et al, 2017).

    laplace: bool
        Whether or not to scale the entries by a Laplace(1). This is not in the original paper.

    neg_idx: None, int
        (Optional) Sets one of the support entries to a negative value.

    random_state: None, int
        Seed for the Laplace scaling.

    Output
    ------
    coef: array-like, shape (n_features, n_responses)
        The coefficient.

    References
    ----------
    Hastie, T., Tibshirani, R. and Tibshirani, R.J., 2017. Extended comparisons of best subset selection, forward stepwise selection, and the lasso. arXiv preprint arXiv:1707.08692.
    """
    rng = check_random_state(random_state)

    if n_responses > 1:
        assert n_nonzero >= n_responses

        coefs = []
        for r in range(n_responses):
            if r == 0:
                neg_idx = None
            else:
                neg_idx = r

            coefs.append(get_sparse_coef(n_features=n_features,
                                         n_nonzero=n_nonzero,
                                         n_responses=1,
                                         beta_type=beta_type,
                                         neg_idx=neg_idx,
                                         laplace=laplace,
                                         random_state=rng))

        return np.vstack(coefs).T

    assert n_nonzero <= n_features
    if neg_idx is not None:
        assert neg_idx < n_nonzero

    if beta_type == 1:
        # roughly equally spaced 1s
        coef = np.zeros(n_features)

        support_idxs = np.linspace(start=0, stop=n_features,
                                   num=n_nonzero,
                                   dtype=int, endpoint=False)

        coef[support_idxs] = 1

    elif beta_type == 2:
        # 1s at beginning
        coef = np.zeros(n_features)
        coef[0:n_nonzero] = 1

        support_idxs = np.arange(n_nonzero)

    elif beta_type == 3:
        # first entries linearly decaying from 10 to 0.5, others zero
        coef = np.zeros(n_features)

        values = np.linspace(start=10, stop=0.5,
                             num=n_nonzero)
        coef[0:n_nonzero] = values

        support_idxs = np.arange(n_nonzero)

    elif beta_type == 4:
        raise NotImplementedError

    elif beta_type == 5:
        # first entries equal to 1, rest exponentially decaying
        n_decay = n_features - n_nonzero
        decay = 0.5 ** np.arange(1, n_decay + 1)
        coef = np.concatenate([np.ones(n_nonzero), decay])

        support_idxs = np.arange(n_nonzero)

    # scale entries by a Laplace
    if laplace:
        scaling = rng.laplace(scale=1, size=n_features)
        coef *= scaling

    # possibly set one of the entries to a negative value
    elif neg_idx is not None:
        coef[support_idxs[neg_idx]] *= -1

    return coef


def get_cov(n_features=10, cov='ar', corr=0.35):
    """
    Sets up the covariance matrix.

    Parameters
    ----------
    n_features: int
        Number of features.

    cov: str
        The covariance matrix of the X data. Must be one of ['ident', 'tot', 'ar']. If cov == 'ident' then we use the identity. If cov == 'ar' then Sigma_{ij} = corr **|i-j| follows an autoregression process. If cov == 'tot' then  (1 - corr) * I + corr * 11^T.

    corr: float
        How correlated the data are.

    Output
    ------
    X: array-like, (n_samples, n_features)
        The sampled
    """

    assert cov in ['ident', 'tot', 'ar']

    # sample X data
    if cov == 'identy':
        cov = np.eye(n_features)

    elif cov == 'tot':
        cov = (1 - corr) * np.eye(n_features) + \
            corr * np.ones((n_features, n_features))

    elif cov == 'ar':
        cov = np.array([[abs(i - j) for i in range(n_features)]
                        for j in range(n_features)])

        cov = corr ** cov

    return cov


def coef_cov_quad_form(coef, cov):
    """
    Computes the quadratic form coef.T @ cov @ coef. Handles the case when coef is is a matrix by returning np.trace(coef.T @ cov @ coef).
    """
    ct_Sigma_c = coef.T @ cov @ coef

    # TODO: is this how we want to handle the SNR?
    if coef.ndim >= 2:
        return np.trace(ct_Sigma_c) / coef.shape[1]
    else:
        return ct_Sigma_c
