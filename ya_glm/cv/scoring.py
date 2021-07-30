from sklearn.metrics import accuracy_score, roc_auc_score, \
    balanced_accuracy_score, f1_score, precision_score, recall_score, \
    median_absolute_error, log_loss

import numpy as np


class Scorer(object):
    def __init__(self, scorer, level=2, default=None):
        self.scorer = scorer
        self.level = level
        self.default = default

    def __call__(self, est, X, y, sample_weight=None):
        return self.scorer(est=est, X=X, y=y, sample_weight=sample_weight,
                           level=self.level)


def get_n_nonzero(coef, zero_tol=1e-8):
    """
    Counts the number of non-zero entries in a coefficient.

    Parameters
    ----------
    core: array-like
        The coefficient.

    zero_tol: float

    """
    return (abs(np.array(coef).reshape(-1)) > zero_tol).sum()


def get_binary_clf_scores(y_true, y_pred, y_score=None,
                          sample_weight=None, level=1):
    """
    Scores a binary classifiers.

    Parameters
    ----------
    y_true: array-like, (n_samples, )
        The ground truth labels.

    y_pred: array-like, (n_samples, )
        The predicted labels.

    y_score: array-like, (n_samples, )
        The predcited scores (e.g. the probabilities)

    sample_weight: array-like shape (n_samples,)
        Sample weights.

    level: int
        How much data to return.

    Output
    ------
    out: dict
        The scores.
    """
    out = {}

    out['accuracy'] = accuracy_score(y_pred=y_pred, y_true=y_true,
                                     sample_weight=sample_weight)

    if y_score is not None:
        out['roc_auc'] = roc_auc_score(y_true=y_true, y_score=y_score,
                                       sample_weight=sample_weight)

        if level >= 2:
            out['log_loss'] = log_loss()

    if level >= 2:
        out['balanced_accuracy'] = \
            balanced_accuracy_score(y_true=y_true, y_pred=y_pred,
                                    sample_weight=sample_weight)

        out['f1'] = f1_score(y_true=y_true, y_pred=y_pred,
                             sample_weight=sample_weight)

        out['precision'] = precision_score(y_true=y_true, y_pred=y_pred,
                                           sample_weight=sample_weight)

        out['recall'] = recall_score(y_true=y_true, y_pred=y_pred,
                                     sample_weight=sample_weight)

    return out


def additional_regression_data(y_true, y_pred, coef, sample_weight=None):
    out = {}
    out['median_abs_err'] = \
        median_absolute_error(y_true=y_true, y_pred=y_pred,
                              sample_weight=sample_weight)

    out['n_nonzero'] = get_n_nonzero(coef, zero_tol=1e-8)
    out['l2_norm'] = np.linalg.norm(coef.reshape(-1))

    return out
