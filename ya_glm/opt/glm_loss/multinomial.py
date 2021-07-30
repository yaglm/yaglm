import numpy as np
from scipy.special import logsumexp
from scipy.sparse import diags

from ya_glm.opt.glm_loss.base import GlmMultiResp
from ya_glm.opt.utils import safe_entrywise_mult
from ya_glm.opt.glm_loss.utils import safe_covar_mat_op_norm
from ya_glm.class_weight import get_sample_weight_balanced_classes


def sample_losses(z, y):
    """
    Negative log probs
    """
    bots = logsumexp(z, axis=1)
    tops = np.array(safe_entrywise_mult(y, z).sum(axis=1)).ravel()
    return bots - tops


def sample_grads(z, y):
    log_probs = z - logsumexp(z, axis=1)[:, np.newaxis]
    probs = np.exp(log_probs)
    return np.array(probs - y)


def compute_lip(X, fit_intercept=True, sample_weight=None):
    # TODO: I think this is right but double check
    op_norm = safe_covar_mat_op_norm(X=X,
                                     fit_intercept=fit_intercept,
                                     sample_weight=sample_weight)

    return (1 / X.shape[0]) * op_norm ** 2


def combine_weights(y, sample_weight=None, class_weight=None):
    if class_weight is not None:
        raise NotImplementedError
        # TODO: need to add this

    return sample_weight


class Multinomial(GlmMultiResp):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)
    compute_lip = staticmethod(compute_lip)

    def __init__(self, X, y, fit_intercept=True, sample_weight=None,
                 balence_classes=False):

        if balence_classes:
            sample_weight = \
                 get_sample_weight_balanced_classes(y=y,
                                                    sample_weight=sample_weight)

        super().__init__(X=X, y=y, fit_intercept=fit_intercept,
                         sample_weight=sample_weight)

    def intercept_at_coef_eq0(self):
        # double check for weighted case
        if self.sample_weight is None:
            return np.array(self.y.mean(axis=0)).ravel()

        else:
            col_sums = (diags(self.sample_weight) @ self.y).sum(axis=0)
            return np.array(col_sums).ravel() / self.X.shape[0]
