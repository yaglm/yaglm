from ya_glm.autoassign import autoassign

from ya_glm.Glm import Glm
from ya_glm.GlmCV import GlmCVSinglePen
from ya_glm.cv.CVPath import CVPathMixin

from ya_glm.glm_pen_max_ridge import ridge_max


class GlmRidge(Glm):

    @autoassign
    def __init__(self, pen_val=1, weights=None, tikhonov=None, **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):

        if self.weights is not None and self.tikhonov is not None:
            raise ValueError("Both weigths and tikhonov cannot both be provided")

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'L2_pen': self.pen_val,
                'L2_weights': self.weights,

                'tikhonov': self.tikhonov
                }

    def _get_pen_val_max_from_pro(self, X, y):

        if self.tikhonov is None:
            return ridge_max(X=X, y=y,
                             fit_intercept=self.fit_intercept,
                             model_type=self._model_type,
                             weights=self.weights,
                             targ_ubd=1,
                             norm_by_dim=True)

        else:
            raise NotImplementedError


class GlmRidgeCVPath(CVPathMixin, GlmCVSinglePen):
    @autoassign
    def __init__(self, weights=None,  tikhonov=None, **kws):
        GlmCVSinglePen.__init__(self, **kws)

    def _get_base_est_params(self):
        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'weights': self.weights,
                'tikhonov': self.tikhonov
                }

    def _get_solve_path_kws(self):

        if self.weights is not None and self.tikhonov is not None:
            raise ValueError("Both weigths and tikhonov cannot both be provided")

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'L2_weights': self.weights,
                'tikhonov': self.tikhonov,

                'L2_pen_seq': self.pen_val_seq_
                }
