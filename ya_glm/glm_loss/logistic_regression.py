from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, column_or_1d
from sklearn.metrics import log_loss
from scipy.special import expit
import numpy as np
from textwrap import dedent

from ya_glm.cv.scoring import get_n_nonzero, score_binary_clf, Scorer


class LogRegMixin(LinearClassifierMixin):

    is_multi_resp = False

    _loss_descr = dedent("""
    Logistic regression with loss function L(z, y) = y log(sig(z)) + (1 - y) log(1 - sig(z))

    where the sigmoid function is sig(z) = 1 / (1 + e^{-z})
    """)

    def get_loss_info(self):
        loss_type = 'log_reg'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_log_reg(y, copy=copy, check_input=True)

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,)
        """
        return self._decision_function(X)

    def predict_proba(self, X):
        """
        """
        check_is_fitted(self)
        return expit(self.decision_function(X))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


def process_y_log_reg(y, copy=True, check_input=True):
    """
    Ensures y is binary.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, out

    y: array-like, shape (n_samples, )
        The possibly mean centered responses.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['y_offset']: float
            The response mean.
    """

    if check_input:
        y = check_array(y, copy=copy, ensure_2d=False)
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)

    elif copy:
        y = y.copy(order='K')
    y = y.reshape(-1)

    enc = LabelEncoder()
    y_ind = enc.fit_transform(y)

    # this class is for binary logistic regression
    assert len(enc.classes_) == 2

    pre_pro_out = {'classes': enc.classes_}

    return y_ind, pre_pro_out


class LogRegScorer(Scorer):
    def __init__(self, verbosity=1, default='roc_auc'):
        super().__init__(scorer=score_log_reg,
                         verbosity=verbosity,
                         default=default)


def score_log_reg(est, X, y, verbosity=1):
    """
    Scores a fitted logistic regression model.

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features)
        The test X data.

    y_true: array-like, shape (n_samples, )
        The true responses.

    est:
        The fit estimator

    verbosity: int
        How much data to return.

    Output
    ------
    scores: dict
        Conatins the scores.
    """
    # get predictions
    y_pred = est.predict(X)
    y_score = est.predict_proba(X)

    # get binary classification scores
    out = score_binary_clf(y_pred=y_pred, y_true=y,
                           y_score=y_score, verbosity=verbosity)

    if verbosity >= 1:
        out['log_loss'] = log_loss(y_true=y, y_pred=y_score)
        out['n_nonzero'] = get_n_nonzero(est.coef_, zero_tol=1e-8)

    return out
