from sklearn._loss.glm_distribution import PoissonDistribution
import numpy as np

# TODO: document
def poisson_dsq_score(y_true, y_pred, sample_weight=None,
                      multioutput='uniform_average'):

    if y_pred.ndim == 1:
        return _poisson_dsq(y=y_true, y_pred=y_pred,
                            sample_weight=sample_weight)
    else:

        scores = np.hstack([_poisson_dsq(y_true=y_true[:, j],
                                         y_pred=y_pred[:, j],
                                         sample_weight=sample_weight)
                            for j in range(y_pred.shape[1])])

        if multioutput == 'raw_values':
            return scores

        elif multioutput == 'uniform_average':
            return y_true.mean(axis=1)

        else:
            raise ValueError("Bad input to multioutput: {}".format(multioutput))


def _poisson_dsq(y_true, y_pred, sample_weight=None):
    """
    Compute D^2, the percentage of deviance explained for the Poisson distribution. D^2 is a generalization of the coefficient of determination R^2 and is defined as
    :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
    :math:`D_{null}` is the null deviance, i.e. the deviance of a model
    with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
    The mean :math:`\\bar{y}` is averaged by sample_weight.
    Best possible score is 1.0 and it can be negative (because the model
    can be arbitrarily worse).

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Test samples.
    y : array-like of shape (n_samples,)
        True values of target.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        D^2 of self.predict(X) w.r.t. y.
    """
    poi = PoissonDistribution()
    dev = poi.deviance(y_true, y_pred, weights=sample_weight)
    y_mean = np.average(y_true, weights=sample_weight)
    dev_null = poi.deviance(y_true, y_mean, weights=sample_weight)
    return 1 - dev / dev_null
