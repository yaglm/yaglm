import numpy as np

from yaglm.metrics.base import Scorer
from yaglm.autoassign import autoassign

from yaglm.config.penalty import Lasso
from yaglm.utils import count_support
from yaglm.extmath import log_binom


class InfoCriteria(Scorer):
    """
    Computes information criteria for GLM model selection. Note this returns the negative of the information criterion such that larger values mean indicate a better fit.

    Parameters
    ----------
    crit: str
        Which information criteria to use. Must be one of ['aic', 'bic', 'ebic'].

    gamma: float, str
        The gamma argument to ebic()

    zero_tol: float, str
        The zero tolerance for support counting. See yaglm.utils.count_support()
    """
    @autoassign
    def __init__(self, crit='ebic', gamma='default', zero_tol=1e-6): pass

    @property
    def name(self):
        return self.crit

    # TODO: currently ignores sample_weight
    def __call__(self, estimator, X, y, sample_weight=None, offsets=None):
        """
        Returns the negative information criteria.

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
            The negative information criteria score so that larger values indicate better model fit.
        """

        if sample_weight is not None:
            raise NotImplementedError("TODO: add")

        if offsets is not None:
            raise NotImplementedError("Think through")

        # formatting
        if not isinstance(estimator.fit_penalty_, Lasso):
            raise NotImplementedError("Information criteria is currently only"
                                      " supported for entrywise sparse penalties.")

        # compute data log-likelihood
        log_lik = estimator.sample_log_liks(X=X, y=y, offsets=offsets).sum()

        n_samples = X.shape[0]

        if self.crit in ['aic', 'bic']:
            dof = estimator.inferencer_.dof_
            if dof is None:
                raise NotImplementedError("The estimator does not currently"
                                          "support estimating the degrees of"
                                          " freedom.")

            if self.crit == 'aic':
                return -aic(log_lik=log_lik, n_samples=n_samples, dof=dof)
            elif self.crit == 'bic':
                return -bic(log_lik=log_lik, n_samples=n_samples, dof=dof)

        elif self.crit == 'ebic':

            n_support = count_support(estimator.coef_, zero_tol=self.zero_tol)
            n_features = estimator.inferencer_.X_shape_[1]

            return -ebic(log_lik=log_lik,
                         n_samples=n_samples, n_features=n_features,
                         n_support=n_support,
                         gamma=self.gamma,
                         fit_intercept=estimator.fit_intercept)

        else:
            raise NotImplementedError("crit must be on of "
                                      "['aic', 'bic', 'ebic'], "
                                      " not {}".format(self.crit))


def bic(log_lik, n_samples, dof):
    """
    Calculates the Bayesian Information Criterion.

    Parameters
    ----------
    log_lik: float
        The observed data log-likelihood.

    n_samples: int
        Number of samples.

    dof: int
        Number of degrees of freedom.

    Output
    ------
    aic: float
    """
    return - 2 * log_lik + np.log(n_samples) * dof


def aic(log_lik, n_samples, dof):
    """
    Calculates the Akaike Information Criterion.

    Parameters
    ----------
    log_lik: float
        The observed data log-likelihood.

    n_samples: int
        Number of samples.

    dof: int
        Number of degrees of freedom.

    Output
    ------
    bic: float
    """
    return - 2 * log_lik + 2 * dof


# TODO: how to generalize this for more general DoF estimates. Both for the formula and the default gamma.
def ebic(log_lik, n_samples, n_features, n_support, gamma='default',
         fit_intercept=True):
    """
    Calculates the Extended Bayesian Information Criterion defined as

    -2 log(Lik) + n_support * log(n_samples) + 2 * gamma log(|model_space|)

    where |model_space| = (n_features choose n_support).

    Parameters
    ----------
    log_lik: float
        The observed data log-likelihood.

    n_samples: int
        Number of samples.

    n_features: int
        Number of features.

    n_support: int
        Number of non-zero coefficient elements.

    gamma: str or float
        If a number, must be between 0 and 1 inclusive. If gamma='default' then we use gamma = 1 - 0.5 * log(n_samples) / log(n_features) as suggested in (Chen and Chen, 2008).

    fit_intercept: bool
        Whether or not an intercept was included in the model.

    Output
    ------
    ebic: float

    References
    ----------
    Chen, J. and Chen, Z., 2008. Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), pp.759-771.
    """

    # agument n_features if there was an intercept
    if fit_intercept:
        n_features = n_features + 1
        n_support = n_support + 1

    # maybe compute default
    if gamma == 'default':
        # default formula from Section 5 of (Chen and Chen, 2008)
        gamma = 1 - 0.5 * (np.log(n_samples) / np.log(n_features))
        gamma = np.clip(gamma, a_min=0, a_max=1)

    # check gamma
    assert gamma >= 0 and gamma <= 1, "Gamma should be in [0, 1]"

    # log of model space log (n_features choose n_support)
    log_model_size = log_binom(n=n_features, k=n_support)

    return bic(log_lik=log_lik, n_samples=n_samples, dof=n_support) + \
        2 * gamma * log_model_size
