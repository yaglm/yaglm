from sklearn.metrics import accuracy_score, roc_auc_score, \
    balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np


class Scorer(object):
    def __init__(self, scorer, verbosity=2, default=None):
        self.scorer = scorer
        self.verbosity = verbosity
        self.default = default

    def __call__(self, est, X, y):
        return self.scorer(est, X, y, verbosity=self.verbosity)


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


def score_binary_clf(y_true, y_pred, y_score=None, verbosity=2):
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

    verbosity: int
        How much data to return.

    Output
    ------
    out: dict
        The scores.
    """
    out = {}

    out['accuracy'] = accuracy_score(y_pred=y_pred, y_true=y_true)

    if y_score is not None:
        out['roc_auc'] = roc_auc_score(y_true=y_true, y_score=y_score)

    if verbosity >= 2:
        out['balanced_accuracy'] = balanced_accuracy_score(y_true=y_true,
                                                           y_pred=y_pred)
        out['f1'] = f1_score(y_true=y_true, y_pred=y_pred)
        out['precision'] = precision_score(y_true=y_true, y_pred=y_pred)
        out['recall'] = recall_score(y_true=y_true, y_pred=y_pred)

    return out
