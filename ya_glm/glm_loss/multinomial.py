import numpy as np
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import softmax
from textwrap import dedent


class MultinomialMixin(LinearClassifierMixin):

    is_multi_resp = True

    _loss_descr = dedent("""
    Multinomial regression with loss function L(y, z) = - sum_{k=1}^{n_classes} y_k log(soft(z)_k) where
    soft(z)_k = e^{z_k} / sum_{j=1}^{n_classes} e^{z_j}
    """)

    def get_loss_info(self):
        loss_type = 'multinomial'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_multinomial(y, check_input=True)

    def decision_function(self, X):
        """

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape (n_samples, n_classes)
        """
        return self._decision_function(X)

    def predict_proba(self, X):
        """
        """
        check_is_fitted(self)
        return softmax(self.decision_function(X))

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

        Output
        -------
        log_provs : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


def process_y_multinomial(y, copy=True, check_input=True):
    """
    Converts y to indicator vectors.

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
    y_ind, out

    y: array-like, shape (n_samples, n_classes)
        The indicator vectors.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['y_offset']: float
            The response mean.
    """

    if check_input:
        y = check_array(y, copy=copy, ensure_2d=False)
        check_classification_targets(y)

    elif copy:
        y = y.copy(order='K')

    lb = LabelBinarizer(sparse_output=True)
    y_ind = lb.fit_transform(y)

    pre_pro_out = {'classes': lb.classes_}

    return y_ind, pre_pro_out
