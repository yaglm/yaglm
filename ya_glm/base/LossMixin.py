import numpy as np
from sklearn.metrics._regression import r2_score
from sklearn.metrics._classification import accuracy_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from textwrap import dedent

from scipy.special import expit

from ya_glm.metrics import poisson_dsq_score

_avail_loss_funcs = ['lin_reg', 'huber', 'quantile',
                     'log_reg', 'multinomial',
                     'poisson']


class LossMixin:

    _params_descr = dedent("""
        loss_func: str
            The loss function to use. Must be one of {}.

        loss_kws: dict
            Keyword arguments that specify the loss function.

            loss_func == 'huber':
                loss_kws['knot']: float
                    The value of the huber knot. Must be a positive value.

            loss_func == 'quantile':
                loss_kws['quantile']: float
                    The quantile. Must be bewtween 0 and 1.
        """)

    def _default_loss_kws(self, loss_func):
        if self.loss_func == 'quantile':
            return {'quantile': 0.5}

        elif self.loss_func == 'huber':
            return {'knot': 1.35}

    def _estimator_type(self):
        if self.loss_func in ['log_reg', 'multinomial']:
            return "classifier"
        else:
            return "regressor"

    def _process_y(self, X, y, sample_weight=None, copy=True, check_input=True):

        if self.loss_func in ['lin_reg', 'huber']:
            return process_y_lin_reg(X=X, y=y,
                                     standardize=self.standardize,
                                     sample_weight=sample_weight,
                                     copy=copy, check_input=check_input)

        elif self.loss_func == 'quantile':
            return process_y_lin_reg(X=X, y=y,
                                     # never center for quantile!
                                     standardize=False,
                                     copy=copy, check_input=check_input)

        elif self.loss_func == 'poisson':
            return process_y_poisson(X=X, y=y,
                                     copy=copy, check_input=check_input)

        elif self.loss_func == 'log_reg':
            return process_y_log_reg(X=X, y=y,
                                     copy=copy,
                                     check_input=check_input)

        elif self.loss_func == 'multinomial':
            return process_y_multinomial(X=X, y=y,
                                         copy=copy,
                                         check_input=check_input)

    def predict(self, X):
        z = self.decision_function(X)

        if self.loss_func in ['lin_reg', 'huber', 'quantile']:
            return z

        elif self.loss_func == 'poisson':
            return np.exp(z)

        elif self.loss_func in ['log_reg', 'multinomial']:

            scores = self.decision_function(X)
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if self.loss_func in ['lin_reg', 'huber', 'quantile', 'poisson']:
            return r2_score(y_true=y, y_pred=y_pred,
                            sample_weight=sample_weight)

        elif self.loss_func == 'poisson':
            return poisson_dsq_score(y_true=y, y_pred=y_pred,
                                     sample_weight=sample_weight)

        elif self.loss_func in ['log_reg', 'multinomial']:
            return accuracy_score(y_true=y, y_pred=y_pred,
                                  sample_weight=sample_weight)

    def predict_proba(self, X):
        check_is_fitted(self)

        if self.loss_func not in ['log_reg', 'multinomial']:
            raise ValueError("{} does not support predict_proba".
                             format(self.loss_func))

        z = self.decision_function(X)

        if self.loss_func == 'log_reg':
            return expit(z)

        elif self.loss_func == 'multinomial':
            return softmax(z)

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


def basic_y_formatting(X, y, copy=True, check_input=True, clf=False):
    if check_input:
        y = check_array(y, copy=copy, ensure_2d=False)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        if clf:
            check_classification_targets(y)
        else:
            y = y.astype(X.dtype)

    elif copy:
        y = y.copy(order='K')

    return y


def process_y_lin_reg(X, y, standardize=False,
                      sample_weight=None,
                      copy=True, check_input=True):
    """
    Processes and possibly mean center the y data i.e. y - y.mean()

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    standardize: bool
        Whether or not to apply standardization to y.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

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

    y = basic_y_formatting(X, y, copy=copy, check_input=check_input)

    # mean center y
    out = {}
    if standardize:
        out['y_offset'] = np.average(a=y, axis=0, weights=sample_weight)
        y -= out['y_offset']

    return y, out


def process_y_log_reg(X, y,
                      copy=True, check_input=True):

    y = basic_y_formatting(X, y, copy=copy, check_input=check_input, clf=True)

    enc = LabelEncoder()
    y_ind = enc.fit_transform(y)

    if check_input:
        # this class is for binary logistic regression
        assert len(enc.classes_) == 2

    pre_pro_out = {'classes': enc.classes_}
    return y_ind, pre_pro_out


def process_y_multinomial(X, y,
                          copy=True, check_input=True):
    y = basic_y_formatting(X, y, copy=copy, check_input=check_input, clf=True)

    if check_input:
        # make sure y is one dimensional
        assert y.ndim == 1

    lb = LabelBinarizer(sparse_output=True)
    y_ind = lb.fit_transform(y)

    pre_pro_out = {'classes': lb.classes_}
    return y_ind, pre_pro_out


def process_y_poisson(X, y,
                      copy=True, check_input=True):

    y = basic_y_formatting(X, y, copy=copy, check_input=check_input)
    if check_input:
        assert y.min() >= 0

    return y, {}
