import numpy as np
from yaglm.utils import count_support


def est_dof_support(coef, intercept=None, transform=None, zero_tol=1e-6):
    """
    The size of the support of the estimated coefficient (or a transform therof) is sometimes a reasonable estimate for the degrees of freedom e.g. in Lasso, SCAD and some generalized Lasso problems.

    Parameters
    ----------
    coef: array-like
        The estimated coefficient.

    intercept: None, float, array-like
        (Optional) The estimated coefficeint.

    transform: None, callable(coef) -> array-like
        (Optional) The transformation applied to the coefficient e.g. for the generalized Lasso.

    zero_tol: float
        Tolerance for declaring a small value equal to zero. This addresses numerical issues where some solvers may not return exact zeros.

    Output
    ------
    dof: int
        The estimaed number of degrees of freedom. The DoF of the coefficeint is given by either ||coef||_0 or ||transform(coef)||_0

    References
    ----------
    Zou, H., Hastie, T. and Tibshirani, R., 2007. On the “degrees of freedom” of the lasso. The Annals of Statistics, 35(5), pp.2173-2192.

    Park, M.Y. and Hastie, T., 2007. L1‐regularization path algorithm for generalized linear models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(4), pp.659-677.

    Zhang, Y., Li, R. and Tsai, C.L., 2010. Regularization parameter selections via generalized information criterion. Journal of the American Statistical Association, 105(489), pp.312-323.
    """

    # count support of estimated coefficient
    coef = np.array(coef)
    if transform is None:
        n_nonzero_coef = count_support(coef, zero_tol=zero_tol)
    else:
        n_nonzero_coef = count_support(transform(coef), zero_tol=zero_tol)

    # maybe add intercept
    if intercept is not None:
        n_vals_intercept = np.array(intercept).size
    else:
        n_vals_intercept = 0

    DoF = n_nonzero_coef + n_vals_intercept
    return DoF


def est_dof_enet(coef, pen_val, mix_val, X, intercept=None, zero_tol=1e-6):
    
    """
    The size of the support of the estimated coefficient for elastic net at a particular 
    penalty and mixing value.
    
    ElasticNet penalty:

    pen_val * mix_val ||coef||_1 + pen_val * (1 - mix_val) * ||coef||_2^2

    Parameters
    ----------
    coef: array-like
        The estimated coefficient.
        
    pen_val: float,
        current penalty value in the elastic net penalty
    
    mix_val: float,
        current mixing value in the elastic net penalty
        
    X: (n,d)-array,
        design matrix excluding the intercept column

    intercept: None, float, array-like
        (Optional) The estimated coefficeint.

    zero_tol: float
        Tolerance for declaring a small value equal to zero. This addresses numerical issues where some solvers may not return exact zeros.

    Output
    ------
    DoF: int
        The estimaed number of degrees of freedom. The DoF of the coefficeint is given by either ||coef||_0 or ||transform(coef)||_0

    References
    ----------
    Zou, H., Hastie, T. and Tibshirani, R., 2007. On the “degrees of freedom” of the lasso. The Annals of Statistics, 35(5), pp.2173-2192.
    """
    
    # Get the estimated support from the fitted coefficient
    if intercept is not None:
        coef = np.concatenate([[intercept], coef])
        
    support = (abs(coef) > zero_tol)
    
    # tuning parameter attached to the ridge penalty
    lambda_2 = pen_val * (1 - mix_val)
    
    # Get the columns of the design matrix that correspond to the non-zero coef
    if intercept is not None:
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate([ones, X], axis = 1)
    
    X_A = X[:, support].copy()
    
    xtx_li_inv = np.linalg.inv(X_A.T @ X_A + lambda_2 * np.identity(X_A.shape[1]))
    
    DoF = np.trace(X_A @ xtx_li_inv @ X_A.T)
    
    return(DoF)
