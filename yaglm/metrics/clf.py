from sklearn.metrics import accuracy_score, roc_auc_score, \
    balanced_accuracy_score, f1_score, precision_score, recall_score, \
    log_loss


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
