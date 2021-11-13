import numpy as np
from scipy.special import logsumexp
from scipy.sparse import diags

from yaglm.opt.glm_loss.base import GlmMultiResp, GlmInputLoss
from yaglm.opt.utils import safe_entrywise_mult


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


def combine_weights(y, sample_weight=None, class_weight=None):
    if class_weight is not None:
        raise NotImplementedError
        # TODO: need to add this

    return sample_weight


class MultinomialLoss(GlmInputLoss):
    sample_losses = staticmethod(sample_losses)
    sample_grads = staticmethod(sample_grads)

    # TODO: add this
    # sample_proxs = !!!!

    @property
    def is_smooth(self):
        return True

    @property
    def grad_lip(self):
        # TODO: double check this
        return 1 / self.n_samples


class Multinomial(GlmMultiResp):

    GLM_LOSS_CLASS = MultinomialLoss

    def intercept_at_coef_eq0(self):
        # double check for weighted case
        if self.sample_weight is None:
            return np.array(self.y.mean(axis=0)).ravel()

        else:
            col_sums = (diags(self.sample_weight) @ self.y).sum(axis=0)
            return np.array(col_sums).ravel() / self.X.shape[0]
