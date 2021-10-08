from sklearn.metrics import r2_score
from ya_glm.metrics import poisson_dsq_score

from ya_glm.cv.scoring import additional_regression_data, \
    get_binary_clf_scores, Scorer


class LinRegScorer(Scorer):
    """
    Custom scoring metrics for linear regression problems.
    """
    def __init__(self, level=1, default='r2'):
        super().__init__(scorer=score_lin_reg,
                         level=level,
                         default=default)


class PoissonScorer(Scorer):
    """
    Custom scoring metrics for poisson regression.
    """
    def __init__(self, level=1, default='dsq'):
        super().__init__(scorer=score_poisson,
                         level=level,
                         default=default)


class BinaryClfScorer(Scorer):
    """
    Custom scoring metrics for binary classification.
    """
    def __init__(self, level=1, default='roc_auc'):
        super().__init__(scorer=score_binary_classifier,
                         level=level,
                         default=default)


class MultiClfScorer(Scorer):
    """
    Custom scoring metrics for multiclass classification.
    """
    def __init__(self, level=1, default='roc_auc'):
        super().__init__(scorer=score_multiclass_classifier,
                         level=level,
                         default=default)


def score_lin_reg(est, X, y, sample_weight=None, level=1):
    """
    Scores a fitted linear regression model.

    Parameters
    -----------
    est:
        The fitted estimator.

    X: array-like, shape (n_samples, n_features)
        The test X data.

    y_true: array-like, shape (n_samples, )
        The true responses.


    sample_weight: array-like shape (n_samples,)
        Sample weights.


    level: int
        How much data to return.
    Output
    ------
    scores: dict
        Containts the scores.
    """

    # get predictions
    y_pred = est.predict(X)

    out = {}
    out['r2'] = r2_score(y_true=y, y_pred=y_pred,
                         sample_weight=sample_weight)

    if level >= 1:
        to_add = additional_regression_data(y_true=y, y_pred=y_pred,
                                            coef=est.coef_,
                                            sample_weight=sample_weight)
        out.update(to_add)

    return out


def score_poisson(est, X, y, sample_weight=None, level=1):
    """
    Scores a fitted linear regression model.

    Parameters
    -----------
    est:
        The fitted estimator.

    X: array-like, shape (n_samples, n_features)
        The test X data.

    y_true: array-like, shape (n_samples, )
        The true responses.

    sample_weight: array-like shape (n_samples,)
        Sample weights.

    level: int
        How much data to return.

    Output
    ------
    scores: dict
        The scores.
    """

    # get predictions
    y_pred = est.predict(X)

    out = {}
    out['dsq'] = poisson_dsq_score(y_true=y, y_pred=y_pred,
                                   sample_weight=sample_weight)

    if level >= 1:
        to_add = additional_regression_data(y_true=y, y_pred=y_pred,
                                            coef=est.coef_,
                                            sample_weight=sample_weight)

        out.update(to_add)

        out['r2'] = r2_score(y_true=y, y_pred=y_pred,
                             sample_weight=sample_weight)

    return out


def score_binary_classifier(est, X, y, sample_weight=None, level=1):
    """
    Scores a fitted binary classifier regression model.

    Parameters
    -----------
    est:
        The fitted estimator.

    X: array-like, shape (n_samples, n_features)
        The test X data.

    y_true: array-like, shape (n_samples, )
        The true responses.

    sample_weight: array-like shape (n_samples,)
        Sample weights.

    level: int
        How much data to return.

    Output
    ------
    scores: dict
        The scores.
    """

    y_pred = est.predict(X)
    y_score = est.predict_proba(X)

    return get_binary_clf_scores(y_true=y, y_pred=y_pred, y_score=y_score,
                                 sample_weight=sample_weight, level=level)


def score_multiclass_classifier(est, X, y, sample_weight=None, level=1):
    # TODO: add this
    raise NotImplementedError
