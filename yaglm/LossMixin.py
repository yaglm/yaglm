import numpy as np
from sklearn.metrics._regression import r2_score
from sklearn.metrics._classification import accuracy_score
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import _check_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from scipy.special import expit

from yaglm.config.loss import get_loss_config
from yaglm.config.base_params import get_base_config
from yaglm.metrics.glm_other import poisson_dsq_score
from yaglm.metrics.glm_log_liks import gaussian, bernoulli, multinomial,\
     poisson
from yaglm.utils import lb_transform_to_indices


class LossMixin:
    """
    Mixin for Glm estimators that handles functionality related to the loss function e.g. predict.
    """

    def _process_y(self, X, y, sample_weight=None, copy=True, check_input=True):

        loss_config = get_base_config(get_loss_config(self.loss))

        if loss_config.name in ['lin_reg', 'huber']:
            return process_y_lin_reg(X=X, y=y,
                                     standardize=self.standardize,
                                     fit_intercept=self.fit_intercept,
                                     sample_weight=sample_weight,
                                     copy=copy, check_input=check_input)

        elif loss_config.name in ['quantile', 'smoothed_quantile']:
            return process_y_lin_reg(X=X, y=y,
                                     sample_weight=sample_weight,
                                     standardize=False,
                                     copy=copy,
                                     check_input=check_input)

        elif loss_config.name == 'poisson':
            return process_y_poisson(X=X, y=y, sample_weight=sample_weight,
                                     copy=copy,
                                     check_input=check_input)

        elif loss_config.name == 'log_reg':
            return process_y_log_reg(X=X, y=y, sample_weight=sample_weight,
                                     class_weight=loss_config.class_weight,
                                     copy=copy,
                                     check_input=check_input
                                     )

        elif loss_config.name == 'multinomial':
            return process_y_multinomial(X=X, y=y, sample_weight=sample_weight,
                                         class_weight=loss_config.class_weight,
                                         copy=copy,
                                         check_input=check_input)

        elif loss_config.name in ['hinge', 'huberized_hinge', 'logistic_hinge']:
            return process_y_hinge(X=X, y=y, sample_weight=sample_weight,
                                   copy=copy,
                                   check_input=check_input
                                   )

    def predict(self, X, offsets=None):
        """
        Returns the predicted values. For regressors this is the E[Y|X], while for classifies this is the predicted class label.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        y_pred: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The predictions.
        """
        if self._estimator_type == 'regressor':
            return self.predict_expected(X, offsets=offsets)

        elif self._estimator_type == 'classifier':

            scores = self.decision_function(X, offsets=offsets)
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]

    def score(self, X, y, sample_weight=None, offsets=None):
        """
        Scores the predictions using the default score strategy e.g. r2_score for linear regression, accuracy for classifiers and D squared for poission.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate test data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The groud truth values for the test samples.

        sample_weight: None or array-like, shape (n_samples,)
            (Optional) Individual weights for each sample.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        score: float
            The score; higher is better.
        """

        y_pred = self.predict(X, offsets=offsets)
        loss_config = get_base_config(get_loss_config(self.loss))

        if loss_config.name in ['lin_reg', 'huber', 'quantile', 'poisson',
                                'smoothed_quantile']:
            return r2_score(y_true=y, y_pred=y_pred,
                            sample_weight=sample_weight)

        elif loss_config.name == 'poisson':
            return poisson_dsq_score(y_true=y, y_pred=y_pred,
                                     sample_weight=sample_weight)

        elif loss_config.name in ['log_reg', 'multinomial',
                                  'hinge', 'huberized_hinge',
                                  'logistic_hinge']:
            return accuracy_score(y_true=y, y_pred=y_pred,
                                  sample_weight=sample_weight)

    def predict_proba(self, X, offsets=None):
        """"
        Predicted class probabilities.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate test data.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        prob: array-like, shape (n_samples, ) or (n_samples, n_classes)
            The probabilities either for class 1 (for logistic regression) or for all the classes (for multinomial.)

        """
        loss_config = get_base_config(get_loss_config(self.loss))
        if loss_config.name not in ['log_reg', 'multinomial']:
            raise ValueError("{} does not support predict_proba".
                             format(loss_config.name))

        return self.predict_expected(X, offsets=offsets)

    def predict_log_proba(self, X, offsets=None):
        """
        Log of the predicted class probabilities.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate test data.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.
        Output
        ------
        prob: array-like, shape (n_samples, ) or (n_samples, n_classes)
            The logs of the probabilities either for class 1 (for logistic regression) or for all the classes (for multinomial.)

        """
        return np.log(self.predict_proba(X, offsets=offsets))

    def predict_expected(self, X, offsets=None):
        """
        Returns E[Y|X], the estimatead expected value of Y given X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate test data.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        expected: array-like (n_samples, n_responses)
            The predicted expected values.
        """
        check_is_fitted(self)

        z = self.decision_function(X, offsets=offsets)
        loss_config = get_base_config(get_loss_config(self.loss))

        if loss_config.name in ['lin_reg', 'huber', 'quantile', 'l2',
                                'smoothed_quantile']:
            return z

        elif loss_config.name == 'poisson':
            return np.exp(z)

        elif loss_config.name == 'log_reg':
            return expit(z)

        elif loss_config.name == 'multinomial':
            return softmax(z)

        else:
            raise NotImplementedError

    def sample_log_liks(self, X, y, offsets=None):
        """
        Returns the sample log-likelihood of the predictions.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate test data.

        y: array-like, shape (n_samples, n_responses)
            The true response test data.

        offsets: None, float, array-like, shape (n_samples, )
            (Optional) The offsets for each sample.

        Output
        ------
        X: array-like, shape (n_samples, )
            The log-likelihood of each sample.
        """
        loss_config = get_base_config(get_loss_config(self.loss))
        y_pred = self.predict_expected(X, offsets=offsets)

        # linear regression
        if loss_config.name == 'lin_reg':

            # get noise estimate
            if hasattr(self, 'inferencer_') \
                    and hasattr(self.inferencer_, 'scale_') \
                    and self.inferencer_.scale_ is not None:

                scale = self.inferencer_.scale_

            else:
                raise RuntimeError("No sigma estimate was found; linear "
                                   "regression requires a noise standard "
                                   "deviation estimate to compute the "
                                   "likelihood.")

            return gaussian(y_pred=y_pred,
                            y_true=y,
                            scale=scale)

        elif loss_config.name == 'poisson':
            return poisson(y_pred=y_pred, y_true=y)

        elif loss_config.name == 'log_reg':

            # converty y to binary labels
            y_true = self.label_encoder_.transform(y)

            return bernoulli(y_pred=y_pred,
                             y_true=y_true)

        elif loss_config.name == 'multinomial':
            # convert y to class indices
            y_true_idxs = lb_transform_to_indices(lb=self.label_binarizer_,
                                                  y=y)

            return multinomial(y_pred=y_pred, y_true=y_true_idxs)

        else:
            raise NotImplementedError("Cannot compute sample log liklihoods"
                                      "for {}.".format(loss_config.name))


def basic_y_formatting(X, y, copy=True, check_input=True, is_clf=False,
                       multi_output=True):

    if copy:
        y = y.copy(order='K')

    if check_input:
        y = _check_y(y, multi_output=multi_output, y_numeric=not is_clf)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        if is_clf:
            check_classification_targets(y)
        else:
            y = y.astype(X.dtype)

    return y


def process_y_lin_reg(X, y,
                      fit_intercept=True,
                      standardize=False,
                      sample_weight=None,
                      copy=True, check_input=True):
    """
    Processes and possibly mean center the y data i.e. y - y.mean().

    If standardize=True, fit_intercept=True we mean center y.
    If standardize=False, fit_intercept=False we do not mean center y

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    fit_intercept: bool
        Whether or not we fit an intercept.

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
    y, sample_weight, out

    y: array-like, shape (n_samples, )
        The possibly mean centered responses.

    sample_weight: None or array-like,  shape (n_samples,)
        The original sample weights.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['y_offset']: float
            The response mean.
    """

    y = basic_y_formatting(X, y, copy=copy, check_input=check_input,
                           multi_output=True)

    # mean center y
    out = {}
    if standardize and fit_intercept:
        out['y_offset'] = np.average(a=y, axis=0, weights=sample_weight)
        y -= out['y_offset']

    return y, sample_weight, out


def process_y_log_reg(X, y, sample_weight=None, class_weight=None,
                      copy=True, check_input=True):
    """
    Binarize the y labels.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    sample_weight: None, array-like (n_samples, )
        The sample weights.

    class_weight: None, 'balanced', dict
        The class weights.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, sample_weight, out

    y: array-like, shape (n_samples, )
        The binary class labels

    sample_weight: None or array-like,  shape (n_samples,)
        The sample weights possibly adjusted by the class weights.

    out: dict
        The pre-processesing output data.
        out['classes']: array-like
            The class labels.
    """
    y = basic_y_formatting(X, y, copy=copy, check_input=check_input,
                           is_clf=True, multi_output=False)

    enc = LabelEncoder()
    y_ind = enc.fit_transform(y)

    # adjust sample weights by class weights
    if class_weight is not None:

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        # compute class weights
        class_weight_vect = compute_class_weight(class_weight=class_weight,
                                                 classes=enc.classes_,
                                                 y=y)

        # multiply origianl sample weights by the class weights
        sample_weight *= class_weight_vect[y_ind]

    if check_input:
        # this class is for binary logistic regression
        assert len(enc.classes_) == 2

    pre_pro_out = {'label_encoder': enc}

    return y_ind, sample_weight, pre_pro_out


def process_y_hinge(X, y, sample_weight=None, class_weight=None,
                    copy=True, check_input=True):
    """
    Converts the labels to +/- 1 for the hinge loss. Note the class labels are determined by sorting order e.g. the "larger" class is the positive class.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    sample_weight: None, array-like (n_samples, )
        The sample weights.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y_pm1, sample_weight, out

    y_pm1: array-like, shape (n_samples, )
        The +/-1 class labels

    sample_weight: None or array-like,  shape (n_samples,)
        The sample weights possibly adjusted by the class weights.

    out: dict
        The pre-processesing output data.
        out['classes']: array-like
            The class labels.
    """

    y = basic_y_formatting(X, y, copy=copy, check_input=check_input,
                           is_clf=True, multi_output=True)

    enc = LabelEncoder()
    y_ind = enc.fit_transform(y)
    y_pm1 = 2 * y_ind - 1

    # adjust sample weights by class weights
    if class_weight is not None:

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        # compute class weights
        class_weight_vect = compute_class_weight(class_weight=class_weight,
                                                 classes=enc.classes_,
                                                 y=y)

        # multiply origianl sample weights by the class weights
        sample_weight *= class_weight_vect[y_ind]

    if check_input:
        # we allow hinge loss to have 1 class!
        assert len(enc.classes_) in [1, 2]

    pre_pro_out = {'label_encoder': enc}

    return y_pm1, sample_weight, pre_pro_out


def pos_neg_encode(enc, y):
    """
    Encodes y to +/-1

    Parameters
    ----------
    enc: LabelEncoder
        The label encoder (binarizes the labels)

    y: array-like, shape (n_samples, )
        The original cass labels

    Output
    ------
    y_pm1: array-like, shape (n_samples, )
        The +/-1 labels
    """
    assert isinstance(enc, LabelEncoder)
    return 2 * enc.transform(y) - 1


def process_y_multinomial(X, y, sample_weight=None, class_weight=None,
                          copy=True, check_input=True):
    """
    Create dummy variables for the y labels.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    sample_weight: None, array-like (n_samples, )
        The sample weights.

    class_weight: None, 'balanced', dict
        The class weights.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, sample_weight, out

    y: array-like, shape (n_samples, n_classes)
        The indicator variables.

    sample_weight: None or array-like,  shape (n_samples,)
        The sample weights possibly adjusted by the class weights.

    out: dict
        The pre-processesing output data.
        out['classes']: array-like
            The class labels.
    """
    y = basic_y_formatting(X, y, copy=copy, check_input=check_input,
                           is_clf=True, multi_output=False)

    if check_input:
        # make sure y is one dimensional
        assert y.ndim == 1

    # convert y to dummy vectors
    lb = LabelBinarizer(sparse_output=True)
    y_dummy = lb.fit_transform(y)

    # adjust sample weights by class weights
    if class_weight is not None:

        # get labels 0, ...., n_classes - 1
        enc = LabelEncoder()
        y_enc = enc.fit_transform(y)

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        # compute class weights
        class_weight_vect = compute_class_weight(class_weight=class_weight,
                                                 classes=enc.classes_,
                                                 y=y)

        # multiply origianl sample weights by the class weights
        sample_weight *= class_weight_vect[y_enc]

    pre_pro_out = {'label_binarizer': lb}
    return y_dummy, sample_weight, pre_pro_out


def process_y_poisson(X, y, sample_weight=None, copy=True, check_input=True):
    """
    Ensures the y data are positive.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    sample_weight: None, array-like (n_samples, )
        The sample weights.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, sample_weight, {}

    y: array-like, shape (n_samples, )
        The y data

    sample_weight: None, array-like (n_samples, )
        The original sample weights.
    sample_weight: None or array-like,  shape (n_samples,)
        The original sample weights.
    """
    y = basic_y_formatting(X, y, copy=copy, check_input=check_input,
                           multi_output=True)
    if check_input:
        assert y.min() >= 0

    return y, sample_weight, {}
