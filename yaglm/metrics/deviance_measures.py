import numpy as np
from scipy.special import betainc, gamma

#########################################################
# Deviance measures
#########################################################


def normal_deviance(y, z_hat, scale=1, sample_weight=None):
    """
    Computes the (possibly scaled) deviance for a normal distribution. See Section 5.1 of (Fan et al., 2020).

    Parameters
    ----------
    y: array-like shape (n_samples, )
        The response data.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept_hat.
        
    scale: float
        The noise variance.

    sample_weight: None, array-like shape (n_samples, )
        (Optional) Sample weights.

    Output
    ------
    dev: float
        The deviance value.

    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    # compute scaled deviace for each sample
    sample_devs = (y-z_hat)**2 / scale
    
    # sum the sample deviances!
    if sample_weight is None:
        return sample_devs.sum()
    else:
        return sample_weight.T @ sample_devs


def poisson_deviance(y, z_hat, sample_weight = None):
    """
    Computes the (possibly scaled) deviance for a poisson distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------
    y: array-like shape (n_samples, )
        The response data.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.

    sample_weight: None, array-like shape (n_samples, )
        Optional sample weights

    Output
    ------
    dev: float
        The deviance value.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    d_i2 = y*np.log(y/z_hat) - (y-z_hat)
    
    if sample_weight is None:
        return 2*sum(d_i2)
    else: 
        return 2*(sample_weight.T @ d_i2)


def binomial_deviance(n, y, z_hat, sample_weight = None):
    """
    Computes the (possibly scaled) deviance for a binomial distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------
    n: array-like shape (n_samples, )
        The number of observations from which the response data were drawn from.
    
    y: array-like shape (n_samples, )
        The response data.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.

    sample_weight: None, array-like shape (n_samples, )
        Optional sample weights

    Output
    ------
    dev: float
        The deviance value.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    d_i2 = y*np.log(y/z_hat) + (n-y)*np.log((n-y)/(n-z_hat))
    
    if sample_weight is None:
        return 2*sum(d_i2)
    else: 
        return 2*(sample_weight.T @ d_i2)
    
    
def gamma_deviance(y, z_hat, sample_weight = None):
    """
    Computes the (possibly scaled) deviance for a gamma distribution. See Section 5.1 of (Fan et al., 2020).

    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.

    sample_weight: None, array-like shape (n_samples, )
        Optional sample weights

    Output
    ------
    dev: float
        The deviance value.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    d_i2 = -np.log(y/z_hat) + (y - z_hat)/z_hat

    if sample_weight is None:
        return 2*sum(d_i2)
    else: 
        return 2*(sample_weight.T @ d_i2)
        

#########################################################
# Additional Residuals
#########################################################


def pearson_resid_normal(y, z_hat):
    """
    Computes the Pearson Residual for a normal distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in (-\infty, \infty).

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in (-\infty, \infty).

    Output
    ------
    ||r_P||_2^2: float
        The sum of squared Pearson Residuals for Normally distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """

    pearson_resid = (y - z_hat)

    return sum(pearson_resid**2)


def pearson_resid_poisson(y, z_hat):
    """
    Computes the Pearson Residual for a Poisson distribution. See Section 5.1 of (Fan et al., 2020). 
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in the natural numbers.
            
    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in the natural numbers.

    Output
    ------
    ||r_P||_2^2: float
        The sum of squared Pearson Residuals for Poisson distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    std_z_hat = z_hat**(1/2)
    pearson_resid = (y - z_hat) / std_z_hat
    return sum(pearson_resid**2)


def pearson_resid_binomial(y, z_hat):
    """
    Computes the Pearson Residual for a Binomial distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be between 0 and 1.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be between 0 and 1.

    Output
    ------
    ||r_P||_2^2: float
        The sum of squared Pearson Residuals for Binomially distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    e_theta = z_hat/(1 - z_hat)
    std_z_hat = np.std(e_theta / (1 - e_theta)**2)
    
    pearson_resid = (y - z_hat) / std_z_hat
    return sum(pearson_resid**2)


def pearson_resid_gamma(y, z_hat, alpha):
    """
    Computes the Pearson Residual for a Gamma distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in (0, \infty).

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in (0, \infty).

    Output
    ------
    ||r_P||_2^2: float
        The sum of squared Pearson Residuals for Gamma distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    
    std_z_hat = z_hat/np.sqrt(alpha)
    pearson_resid = (y - z_hat) / std_z_hat
    return sum(pearson_resid**2) 
    
    
def anscombe_resid_normal(y, z_hat):
    """
    Computes the Anscombe Residual for a normal distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in (-\infty, \infty).

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in (-\infty, \infty).

    Output
    ------
    ||r_A||_2^2: float
        The sum of squared Anscombe Residuals for Normally distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    https://v8doc.sas.com/sashtml/insight/chap39/sect57.htm
    """
    
    anscomb_resid = y - z_hat
    return sum(anscomb_resid**2)


def anscombe_resid_poisson(y, z_hat):
    """
    Computes the Anscombe Residual for a Poisson distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in the natural numbers.
            
    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in the natural numbers.

    Output
    ------
    ||r_A||_2^2: float
        The sum of squared Anscombe Residuals for Poisson distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    https://v8doc.sas.com/sashtml/insight/chap39/sect57.htm
    """
    
    anscomb_resid = (3/2) * ((y**(2/3)) * z_hat**(-1/6) - z_hat**(1/2))
    return sum(anscomb_resid**2)


def anscombe_resid_binomial(m, y, z_hat):
    """
    Computes the Anscombe Residual for a Binomial distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be between 0 and 1.

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be between 0 and 1.

    Output
    ------
    ||r_A||_2^2: float
        The sum of squared Anscombe Residuals for Binomially distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    https://v8doc.sas.com/sashtml/insight/chap39/sect57.htm
    """
    
    a = 2/3
    b = 2/3
    gamma_ab = gamma(a + b)
    gamma_a = gamma(a)
    gamma_b = gamma(b)
    
    reg_par = (gamma_a * gamma_b) / gamma_ab
    
    beta_y = reg_par * betainc(gamma_a, gamma_b, y)
    beta_z_hat = reg_par * betainc(gamma_a, gamma_b, z_hat)
    
    denom = (z_hat * (1 - z_hat))**(-1 / 6)
    
    anscomb_resid = np.sqrt(m) * (beta_y - beta_z_hat) / denom
    
    return sum(anscomb_resid**2)


def anscombe_resid_gamma(y, z_hat):
    """
    Computes the Anscombe Residual for a Gamma distribution. See Section 5.1 of (Fan et al., 2020).
    
    Parameters
    ----------    
    y: array-like shape (n_samples, )
        The response data.
        Assumed to be in (0, \infty).

    z_hat: array-like shape (n_samples, )
        The linear predictions z_hat = X @ beta_hat + intercept.
        Assumed to be in (0, \infty).

    Output
    ------
    ||r_A||_2^2: float
        The sum of squared Anscombe Residuals for Normally distributed observations.
        
    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    https://v8doc.sas.com/sashtml/insight/chap39/sect57.htm
    """
    
    anscomb_resid = 3 * ((y / z_hat)**(1/3) - 1)
    return sum(anscomb_resid**2)

#########################################################
# Tuning parameter selection
#########################################################


def gcv_score(fit_measure, n_samples, df):
    """
    Computes the generalized cross-validation score. See Section 5.6 of (Fan et al., 2020).

    Parameters
    ----------    
    fit_measure: float
       A measure of model fit, e.g. deviance, sum of squared Pearson, or Anscombe Residuals.

    n_samples: int
       Number of samples.
       
    df: float
        An estimate of the effective degrees of freedom for the estimated coefficient.

    Output
    ------
    gcv: float
        Generalized cross-validation score.

    References
    ----------
    Fan, J., Li, R., Zhang, C.H. and Zou, H., 2020. Statistical foundations of data science. Chapman and Hall/CRC.
    """
    return (1/n_samples) * fit_measure / (1 - (df/n_samples))**2






















