import numpy as np
from scipy.special import loggamma
from numbers import Number


def gaussian(y_pred, y_true, scale):
    """
    Computes the log-likelihood of each sample for a Gaussian GLM given the predicted expected values.

    Parameters
    ----------
    y_pred: array-like, (n_samples, ) or (n_samples, n_responses)
        The predicted expected values of y.

    y_true: array-like, (n_samples, ) or (n_samples, n_responses)
        The actual y values.

    scale: float or array-like (n_responses, )
        The gaussian standard deviation parameter.

    Output
    ------
    log_liks: array-like, (n_samples, ) or (n_samples, n_responses)
        The sample log-likelihoods
    """
    return _safe_apply_over_responses(y_pred=y_pred,
                                      y_true=y_true,
                                      func=_gaussian,
                                      scale=scale)


def _gaussian(y_pred, y_true, scale):
    """
    Computes the log-likelihood of each sample for a Gaussian GLM given the predicted expected values.
    """
    return - 0.5 * (y_pred - y_true) ** 2 / scale ** 2 \
        - 0.5 * np.log(2 * np.pi * scale**2)


def bernoulli(y_pred, y_true):
    """
    Computes the log-likelihood of each sample for a Bernoulli GLM given the predicted expected values.

    Parameters
    ----------
    y_pred: array-like, (n_samples, )
        The predicted expected values of y.

    y_true: array-like, (n_samples, )
        The true binary class labels (must be 0 or 1).

    Output
    ------
    log_liks: array-like, (n_samples, )
        The sample log-likelihoods
    """

    # make sure these are flat numpy arrays
    #y_pred = np.array(y_pred).reshape(-1)
    # y_true = np.array(y_true).reshape(-1)

    log_liks = np.zeros(len(y_pred))

    ones_mask = y_true == 1

    log_liks[ones_mask] = np.log(y_pred[ones_mask])
    log_liks[~ones_mask] = np.log(1 - y_pred[~ones_mask])
    return log_liks


def multinomial(y_pred, y_true):
    """
    Computes the log-likelihood of each sample for a Multinomial GLM given the predicted expected values.

    Parameters
    ----------
    y_pred: array-like, (n_samples, n_responses)
        The predicted expected values of y.

    y_true: array-like, (n_samples, )
        The actual y values as integer class labels between 0 and n_responses - 1.

    Output
    ------
    log_liks: array-like, (n_samples, )
        The sample log-likelihoods
    """
    n_samples = y_pred.shape[0]
    probs = np.array([y_pred[i, y_true[i]] for i in range(n_samples)])
    return np.log(probs)


def poisson(y_pred, y_true):
    """
    Computes the log-likelihood of each sample for a Poisson GLM given the predicted expected values.

    Parameters
    ----------
    y_pred: array-like, (n_samples, ) or (n_samples, n_responses)
        The predicted expected values of y.

    y_true: array-like, (n_samples, ) or (n_samples, n_responses)
        The actual y values. Must be either 0 or 1.

    Output
    ------
    log_liks: array-like, (n_samples, ) or (n_samples, n_responses)
        The sample log-likelihoods
    """
    return _safe_apply_over_responses(y_pred=y_pred,
                                      y_true=y_true,
                                      func=_poisson)


def _poisson(y_pred, y_true):
    """
    Computes the log-likelihood of each sample for a Poisson GLM given the predicted expected values.
    """
    return y_true * np.log(y_pred) - y_pred - loggamma(y_true)


def _format_y(y):
    """
    Ensures y is a vector
    Parameters
    ----------
    y: array-like, shape (n_samples, n_responses)
        The input array.

    Output
    ------
    y: array-like
        y as a numpy array. This is a vector if n_responses=1.
    """
    y = np.array(y)
    if y.ndim == 2 and y.shape[0] == 1:
        return y.reshape(-1)
    else:
        return y


def _safe_apply_over_responses(y_pred, y_true, func, scale=None):
    """
    Safely compute the sample log-likelhood over a potentially multi-response output.

    Parameters
    ----------
    y_pred: array-like, (n_samples, ) or (n_samples, n_responses)
        The predicted expected value for each sample.

    y_true: array-like, (n_samples, ) or (n_samples, n_responses)
        The true value for each sample.

    func: callable(y_pred, y_true, **scale) -> array-like, (n_samples )
        A function that computes the sample log-liklihood for a 1d response.

    scale
        (Optional) Optional scale parameter.
    """
    y_pred = _format_y(y_pred)
    y_true = _format_y(y_true)

    # multi-response case
    if y_pred.ndim == 2:
        n_responses = y_pred.shape[1]

        # create list of dicts
        if scale is not None:

            # if a number value was input then broadcast it to all responses
            if isinstance(scale, Number):
                scale_lod = [{'scale': scale} for r in range(n_responses)]
            else:
                assert len(scale) == n_responses
                scale_lod = [{'scale': scale[r]} for r in range(n_responses)]

        else:
            scale_lod = [{} for r in range(n_responses)]

        # compute the sample likelihoods for each
        return np.hstack([func(y_pred=y_pred[:, r],
                               y_true=y_true[: r],
                               **scale_lod[r])
                          for r in range(n_responses)])

    else:
        if scale is None:
            return func(y_pred=y_pred, y_true=y_true)
        else:
            return func(y_pred=y_pred, y_true=y_true, scale=scale)
