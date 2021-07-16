import numpy as np
from ya_glm.linalg_utils import smallest_sval


def get_pen_max(X, y, fit_intercept=True,
                weights=None,
                loss_func='lin_reg', loss_kws={},
                targ_ubd=1, norm_by_dim=True):
    """
    Returns a heuristic for the largest reasonable value for the ridge tuning parameter. See linear_regression_max_val documentation for a description.


    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, )
        The training response data.

    fit_intercept: bool
        Whether or not to fit an intercept.

    weights: None, array-like
        Optional L2 weights or Tikhinov regularization matrix.

    loss_func: str
        Which GLM loss function we are fitting.

    loss_kws: dict
        Keyword arguments for loss function.

    targ_ubd: float
        The targeted upper bound.

    norm_by_dim: bool
        Whether the targeted upper bound metric should be normalized by the dimension.
    """

    if loss_func == 'lin_reg':
        return lin_reg_ridge_max(X, y, fit_intercept,
                                 targ_ubd=targ_ubd,
                                 weights=weights,
                                 norm_by_dim=norm_by_dim)

    elif loss_func == 'log_reg':
        return log_reg_ridge_max(X, y, fit_intercept,
                                 targ_ubd=targ_ubd,
                                 weights=weights,
                                 norm_by_dim=norm_by_dim)

    else:
        raise NotImplementedError('{} not supported'.format(loss_func))


def lin_reg_ridge_max(X, y, fit_intercept=True, weights=None,
                      targ_ubd=1, norm_by_dim=True):
    """
    Computes a default value for the largest ridge penalty value to try.
    This value is set such that the norm of the estimated coefficient is small, 
    in particular
    
    ||beta_{pen_defualt}||_* <= targ_ubd
    
    where * is either max or L2.
    
    This bound is based off the formula

    coef_{gamma} = (X^TX + n * gamma I_d)^{-1} X^Ty

    which leads to the inequality

    ||coef_{gamma}||_2 <= ||X^Ty||_2 /(n * gamma + eval_min(X^TX))

    (note the n comes from the fact our ridge loss function is (0.5/n)||Xcoef - y||_2^2)

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data
    
    y: array-like, shape (n_samples, )
    
    fit_intercept: bool
        Whether or not we will fit an intercept.

    weights: None, array-like
        Optional L2 weights or Tikhinov regularization matrix.
        
    targ_ubd: float
        The targeted upper bound.
        
    norm_by_dim: bool
        If we want to normalize by the dimension to give
        ||coef_2||/ sqrt(n_features) <= targ_ubd.
        
    Output
    ------
    pen_val_max_default: float
        The defualt largest ridge penalty value to use.
    
    """
    # TODO: normalize is a bad name here -- change this name

    if weights is not None:
        raise NotImplementedError

    if norm_by_dim:
        targ_ubd = targ_ubd / np.sqrt(X.shape[1])
        
    eval_min = smallest_sval(X) ** 2

    if fit_intercept:
        scaled_prod = np.linalg.norm(X.T @ (y - y.mean())) / targ_ubd
    else:
        scaled_prod = np.linalg.norm(X.T @ y) / targ_ubd

    # rescale by number of samples
    n_samples = X.shape[0]
    scaled_prod /= n_samples
    eval_min /= n_samples

    if scaled_prod <= eval_min:
        # when the coefficeint is alreay very small
        # just return the smallest eigenvalue
        # which is a reasonable scale for the ridge penalty
        return eval_min
    
    else:
        return scaled_prod - eval_min


def log_reg_ridge_max(X, y, fit_intercept=True, weights=None,
                      targ_ubd=1, norm_by_dim=True):
    """
    This is my best guess for a reasonable default largest ridge penalty value for logistic
    regression. Perhaps there are better options and perhaps this deserves thinking through more.

    It is based off the following idea

    Recall the newton update for unpenalized logistic regression
    e.g. see slide 20 of http://personal.psu.edu/jol2/course/stat597e/notes2/logit.pdf

    beta_new = (X^T diag(w) X)^{-1} X^T diag(w) z
    where
    z = X beta_old + diag(w)^{-1}(y - p_old)
    p_old is the old vector of fitted probabilities
    w = p_old (1 - p_old)

    Note this formula looks like linear regression with transformed data X
    X_trans = X diag(sqrt(w)) and z as a response

    Suppose we start the algo from beta_old = 0 (reasonable since we hope the ridge penalty will make the solution small)
    If beta_old = 0 its reasonable to guess p_old = avg(y)

    We now choose the ridge penalty based on the largest default ridge penalty for linear regression of the transformed data.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data

    y: array-like, shape (n_samples, )

    fit_intercept: bool
        Whether or not we will fit an intercept.

    see arguments to linear_regression_max_val for other arguments

    Output
    ------
    pen_val_max_default: float
        The defualt largest ridge penalty value to use.

    """
    if fit_intercept:
        p = np.mean(y)
    else:
        p = 0.5

    w = p * (1 - p)

    X_trans = X * np.sqrt(w)

    z = (y - p) / max(w, np.finfo(float).eps)

    return lin_reg_ridge_max(X=X_trans,
                             y=z,
                             fit_intercept=fit_intercept,
                             weights=weights,
                             targ_ubd=targ_ubd,
                             norm_by_dim=norm_by_dim)
