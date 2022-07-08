"""
This file replaces sklearn.metrics._scorer.py with minor modifications allowing the estimator's decition_fuction, predict, predict_proba calls to accept an 'offsets' argument.
"""

from sklearn.metrics._scorer import _BaseScorer as _sk_BaseScorer
from sklearn.metrics._scorer import _PredictScorer as _sk_PredictScorer
from sklearn.metrics._scorer import _ProbaScorer as _sk_ProbaScorer
from sklearn.metrics._scorer import _ThresholdScorer as _sk_ThresholdScorer

from sklearn.metrics._scorer import _cached_call
from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_regressor

from sklearn.metrics import (
    r2_score,
    median_absolute_error,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    log_loss,
    balanced_accuracy_score,
    explained_variance_score,
    brier_score_loss,
    jaccard_score,
    mean_absolute_percentage_error,
)

from sklearn.metrics.cluster import (
    adjusted_rand_score,
    rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score
)


import numpy as np
import inspect
from functools import partial


def check_accepts_offsets(func):
    'offsets' in inspect.getargspec(func).args


class _BaseScorer(_sk_BaseScorer):

    def __call__(self, estimator, X, y_true, sample_weight=None, offsets=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        offsets : array-like of shape (n_samples,), default=None
            Sample offsets.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(
            partial(_cached_call, None),
            estimator,
            X,
            y_true,
            sample_weight=sample_weight,
            offsets=offsets
        )


class _PredictScorer(_sk_PredictScorer):
    def _score(self, method_caller, estimator, X, y_true,
               sample_weight=None, offsets=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        offsets : array-like of shape (n_samples,), default=None
            Sample offsets.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if offsets is not None:
            assert check_accepts_offsets(estimator.predict)
            y_pred = method_caller(estimator, "predict", X=X, offsets=offsets)
        else:
            y_pred = method_caller(estimator, "predict", X)

        if sample_weight is not None:
            return self._sign * self._score_func(
                y_true, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)


class _ProbaScorer(_sk_ProbaScorer):
    def _score(self, method_caller, clf, X, y,
               sample_weight=None, offsets=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, default=None
            Sample weights.

        offsets : array-like of shape (n_samples,), default=None
            Sample offsets.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if offsets is not None:
            assert check_accepts_offsets(clf.predict_proba)
            y_pred = method_caller(clf, "predict_proba", X=X, offsets=offsets)
        else:
            y_pred = method_caller(clf, "predict_proba", X)

        y_type = type_of_target(y)

        if y_type == "binary" and y_pred.shape[1] <= 2:
            # `y_type` could be equal to "binary" even in a multi-class
            # problem: (when only 2 class are given to `y_true` during scoring)
            # Thus, we need to check for the shape of `y_pred`.
            y_pred = self._select_proba_binary(y_pred, clf.classes_)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)


class _ThresholdScorer(_sk_ThresholdScorer):
    def _score(self, method_caller, clf, X, y,
               sample_weight=None, offsets=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, default=None
            Sample weights.

        offsets : array-like of shape (n_samples,), default=None
            Sample offsets.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_type = type_of_target(y)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if is_regressor(clf):

            if offsets is not None:
                assert check_accepts_offsets(clf.predict)
                y_pred = method_caller(clf, "predict", X=X, offsets=offsets)
            else:
                y_pred = method_caller(clf, "predict", X)

        else:
            try:
                if offsets is not None:
                    assert check_accepts_offsets(clf.decision_function)
                    y_pred = method_caller(clf, "decision_function",
                                           X=X, offsets=offsets)
                else:
                    y_pred = method_caller(clf, "decision_function", X)

                if isinstance(y_pred, list):
                    # For multi-output multi-class estimator
                    y_pred = np.vstack([p for p in y_pred]).T
                elif y_type == "binary" and "pos_label" in self._kwargs:
                    self._check_pos_label(self._kwargs["pos_label"], clf.classes_)
                    if self._kwargs["pos_label"] == clf.classes_[0]:
                        # The implicit positive class of the binary classifier
                        # does not match `pos_label`: we need to invert the
                        # predictions
                        y_pred *= -1

            except (NotImplementedError, AttributeError):

                if offsets is not None:
                    assert check_accepts_offsets(clf.predict_proba)
                    y_pred = method_caller(clf, "predict_proba",
                                           X=X, offsets=offsets)
                else:
                    y_pred = method_caller(clf, "predict_proba", X)

                if y_type == "binary":
                    y_pred = self._select_proba_binary(y_pred, clf.classes_)
                elif isinstance(y_pred, list):
                    y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)


######### Everything below is just copy/paste from sklearn #########

def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.

    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError(
                "%r is not a valid scoring value. "
                "Use sorted(sklearn.metrics.SCORERS.keys()) "
                "to get valid options." % scoring
            )
    else:
        scorer = scoring
    return scorer


def make_scorer(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a score function, such as :func:`~sklearn.metrics.accuracy_score`,
    :func:`~sklearn.metrics.mean_squared_error`,
    :func:`~sklearn.metrics.adjusted_rand_index` or
    :func:`~sklearn.metrics.average_precision`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : bool, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

    needs_threshold : bool, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function` or :term:`predict_proba` when
    :term:`decision_function` is not present.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error, greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(
    mean_squared_log_error, greater_is_better=False
)
neg_mean_absolute_error_scorer = make_scorer(
    mean_absolute_error, greater_is_better=False
)
neg_mean_absolute_percentage_error_scorer = make_scorer(
    mean_absolute_percentage_error, greater_is_better=False
)
neg_median_absolute_error_scorer = make_scorer(
    median_absolute_error, greater_is_better=False
)
neg_root_mean_squared_error_scorer = make_scorer(
    mean_squared_error, greater_is_better=False, squared=False
)
neg_mean_poisson_deviance_scorer = make_scorer(
    mean_poisson_deviance, greater_is_better=False
)

neg_mean_gamma_deviance_scorer = make_scorer(
    mean_gamma_deviance, greater_is_better=False
)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
top_k_accuracy_scorer = make_scorer(
    top_k_accuracy_score, greater_is_better=True, needs_threshold=True
)
roc_auc_scorer = make_scorer(
    roc_auc_score, greater_is_better=True, needs_threshold=True
)
average_precision_scorer = make_scorer(average_precision_score, needs_threshold=True)
roc_auc_ovo_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
roc_auc_ovo_weighted_scorer = make_scorer(
    roc_auc_score, needs_proba=True, multi_class="ovo", average="weighted"
)
roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
roc_auc_ovr_weighted_scorer = make_scorer(
    roc_auc_score, needs_proba=True, multi_class="ovr", average="weighted"
)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
neg_brier_score_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, needs_proba=True
)
brier_score_loss_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, needs_proba=True
)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
rand_scorer = make_scorer(rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(
    explained_variance=explained_variance_scorer,
    r2=r2_scorer,
    max_error=max_error_scorer,
    neg_median_absolute_error=neg_median_absolute_error_scorer,
    neg_mean_absolute_error=neg_mean_absolute_error_scorer,
    neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer,  # noqa
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
    neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
    neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
    neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
    accuracy=accuracy_scorer,
    top_k_accuracy=top_k_accuracy_scorer,
    roc_auc=roc_auc_scorer,
    roc_auc_ovr=roc_auc_ovr_scorer,
    roc_auc_ovo=roc_auc_ovo_scorer,
    roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
    roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
    balanced_accuracy=balanced_accuracy_scorer,
    average_precision=average_precision_scorer,
    neg_log_loss=neg_log_loss_scorer,
    neg_brier_score=neg_brier_score_scorer,
    # Cluster metrics that use supervised evaluation
    adjusted_rand_score=adjusted_rand_scorer,
    rand_score=rand_scorer,
    homogeneity_score=homogeneity_scorer,
    completeness_score=completeness_scorer,
    v_measure_score=v_measure_scorer,
    mutual_info_score=mutual_info_scorer,
    adjusted_mutual_info_score=adjusted_mutual_info_scorer,
    normalized_mutual_info_score=normalized_mutual_info_scorer,
    fowlkes_mallows_score=fowlkes_mallows_scorer,
)


for name, metric in [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
    ("jaccard", jaccard_score),
]:
    SCORERS[name] = make_scorer(metric, average="binary")
    for average in ["macro", "micro", "samples", "weighted"]:
        qualified_name = "{0}_{1}".format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None, average=average)
