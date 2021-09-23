import numpy as np


from ya_glm.metrics.base import MultiScorer
from ya_glm.utils import count_support


# TODO: add multi-response metrics
class FitEval(MultiScorer):
    """
    Measures various fit quantities

    """
    def __init__(self, zero_tol=1e-6): pass

    def __call__(self, estimator, X, y, sample_weight=None):
        """
        Output
        ------
        evals: dict of float
            The fit measures
        """
        coef = estimator.coef_.reshape(-1)

        metrics = {}
        metrics['n_nonzero'] = count_support(coef, zero_tol=self.zero_tol)
        metrics['L1'] = np.linalg.norm(x=coef, ord=1)
        metrics['L2'] = np.linalg.norm(x=coef, ord=2)
        metrics['max'] = abs(coef).max()

        return metrics
