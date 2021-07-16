from sklearn.base import RegressorMixin
from sklearn._loss.glm_distribution import PoissonDistribution
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_array, column_or_1d

import numpy as np


class PoissonRegMixin(RegressorMixin):
    is_multi_resp = False

    def get_loss_info(self):
        loss_type = 'poisson'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_poisson(y, copy=copy, sample_weight=sample_weight,
                                 check_input=True)

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        z = self.decision_function(X)
        return np.exp(z)

    def score(self, X, y, sample_weight=None):
        sample_weight = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)
        return poisson_Dsq(y_pred=y_pred, y=y, sample_weight=sample_weight)


class PoissonRegMultiRespMixin(RegressorMixin):

    is_multi_resp = True

    def get_loss_info(self):
        loss_type = 'poisson_mr'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_poisson_multi_resp(y, sample_weight=sample_weight,
                                            copy=copy, check_input=True)

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        z = self.decision_function(X)
        return np.exp(z)

    def score(self, X, y, sample_weight=None):
        sample_weight = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)

        return sum(poisson_Dsq(y_pred=y_pred[:, k], y=y[:, k],
                               sample_weight=sample_weight)
                   for k in range(y.shape[1]))


def process_y_poisson(y, sample_weight=None, copy=True, check_input=True):

    if check_input:
        y = check_array(y, copy=copy, ensure_2d=False)
        y = column_or_1d(y, warn=True)
        assert y.min() >= 0
    elif copy:
        y = y.copy(order='K')

    y = y.reshape(-1)

    return y, {}


def process_y_poisson_multi_resp(y, sample_weight=None,
                                 copy=True, check_input=True):

    if check_input:
        y = check_array(y, copy=copy, ensure_2d=True)
        assert y.min() >= 0
    elif copy:
        y = y.copy(order='K')

    return y, {}


def poisson_Dsq(y_pred, y, sample_weight=None):
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
    dev = poi.deviance(y, y_pred, weights=sample_weight)
    y_mean = np.average(y, weights=sample_weight)
    dev_null = poi.deviance(y, y_mean, weights=sample_weight)
    return 1 - dev / dev_null
