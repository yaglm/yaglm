from textwrap import dedent

from ya_glm.base.Glm import Glm
from ya_glm.base.GlmCV import GlmCVSinglePen
from ya_glm.cv.CVPath import CVPathMixin
from ya_glm.cv.CVGridSearch import CVGridSearchMixin
from ya_glm.pen_max.ridge import get_pen_max

from ya_glm.init_signature import add_from_classes
from ya_glm.utils import maybe_add
from ya_glm.processing import check_estimator_type


_glm_ridge_params = dedent("""
pen_val: float
    The penalty value.

weights: None, array-like shape (n_featuers, )
    Optional features weights for the ridge peanlty.

tikhonov: None, array-like (K, n_features)
    Optional tikhonov matrix for the ridge penalty. Both tikhonov and weights cannot be provided at the same time.
    """)


class GlmRidge(Glm):

    descr = dedent("""
    Ridge penalty.
    """)

    @add_from_classes(Glm)
    def __init__(self, pen_val=1, weights=None, tikhonov=None): pass

    def _get_solve_kws(self):

        if self.weights is not None and self.tikhonov is not None:
            raise ValueError("Both weigths and tikhonov"
                             "cannot both be provided")

        loss_func, loss_kws = self.get_loss_info()

        kws = {'loss_func': loss_func,
               'loss_kws': loss_kws,

               'fit_intercept': self.fit_intercept,
               **self.opt_kws,

               'ridge_pen': self.pen_val,
               }

        extra_kws = {'ridge_weights': self.weights,
                     'tikhonov': self.tikhonov
                     }

        kws = maybe_add(kws, **extra_kws)

        return kws

    def _get_pen_val_max_from_pro(self, X, y, sample_weight=None):
        loss_func, loss_kws = self.get_loss_info()

        if self.tikhonov is None:
            return get_pen_max(X=X, y=y,
                               fit_intercept=self.fit_intercept,
                               loss_func=loss_func,
                               loss_kws=loss_kws,
                               weights=self.weights,
                               sample_weight=sample_weight,
                               targ_ubd=1,
                               norm_by_dim=True)

        else:
            raise NotImplementedError
            # TODO


class GlmRidgeCVPath(CVPathMixin, GlmCVSinglePen):

    descr = dedent("""
    Tunes the ridge penalty parameter via cross-validation using a path algorithm.
    """)

    def _get_solve_path_kws(self):
        if not hasattr(self, 'pen_val_seq_'):
            raise RuntimeError("pen_val_seq_ has not yet been set")

        kws = self.estimator._get_solve_kws()
        del kws['ridge_pen']
        kws['ridge_pen_seq'] = self.pen_val_seq_
        return kws

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmRidge)


class GlmRidgeCVGridSearch(CVGridSearchMixin, GlmCVSinglePen):

    descr = dedent("""
    Tunes the ridge penalty parameter via cross-validation.
    """)

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmRidge)
