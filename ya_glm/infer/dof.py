import numpy as np
from ya_glm.utils import count_support


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
