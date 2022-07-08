class Scorer:
    """
    Base class for an estimator scoring object.
    """

    def __call__(self, estimator, X, y, sample_weight=None, offsets=None):
        """
        Parameters
        ----------
        estimator: Estimator
            The fit estimator to score.

        X: array-like, shape (n_samples, n_features)
            The covariate data to used for scoring.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data to used for scoring.

        sample_weight: None, array-like (n_samples, )
            (Optional) Sample weight to use for scoring.

        offsets: None, array-like (n_samples, )
            (Optional) Sample offsets.

        Output
        ------
        scores: float
            The scores. For measures of fit larger scores should always indicate better fit.
        """
        raise NotImplementedError("Subclass should overwrite")

    @property
    def name(self):
        """
        Output
        ------
        name: str
            Name of this scoring object.
        """
        raise NotImplementedError("Subclass should overwrite")


class MultiScorer:
    """
    Base class for an estimator scoring object returning multiple scores.

    Parameters
    ----------
    default: str
        Name of the default score to use.

    Attributes
    ----------
    name: str
        Name of this score.
    """
    def __call__(self, estimator, X, y, sample_weight=None, offsets=None):
        """

        Output
        ------
        scores: dict of float
            The scores. For measures of fit larger scores should always indicate better fit.
        """
        raise NotImplementedError("Subclass should overwrite")
