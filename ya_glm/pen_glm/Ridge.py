from ya_glm.Glm import Glm
from ya_glm.GlmCV import GlmCVSinglePen
from ya_glm.cv.CVPath import CVPathMixin

from ya_glm.glm_pen_max_ridge import ridge_max
from ya_glm.add_init_params import add_init_params
from ya_glm.processing import check_estimator_type


class GlmRidge(Glm):

    @add_init_params(Glm)
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
               'ridge_weights': self.weights,

               'tikhonov': self.tikhonov
               }

        return kws

    def _get_pen_val_max_from_pro(self, X, y):
        loss_func, loss_kws = self.get_loss_info()

        if self.tikhonov is None:
            return ridge_max(X=X, y=y,
                             fit_intercept=self.fit_intercept,
                             loss_func=loss_func,
                             loss_kws=loss_kws,
                             weights=self.weights,
                             targ_ubd=1,
                             norm_by_dim=True)

        else:
            raise NotImplementedError
            # TODO


class GlmRidgeCVPath(CVPathMixin, GlmCVSinglePen):

    def _get_solve_path_kws(self):
        if not hasattr(self, 'pen_val_seq_'):
            raise RuntimeError("pen_val_seq_ has not yet been set")

        kws = self.estimator._get_solve_kws()
        del kws['ridge_pen']
        kws['ridge_pen_seq'] = self.pen_val_seq_
        return kws

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmRidge)
