from ya_glm.autoassign import autoassign

from ya_glm.Glm import Glm
from ya_glm.GlmCV import GlmCVSinglePen
from ya_glm.cv.CVPath import CVPathMixin
from ya_glm.glm_pen_max_lasso import nuclear_norm_max


class GlmNuclearNorm(Glm):

    @autoassign
    def __init__(self, pen_val=1, weights=None, **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):
        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'lasso_pen': self.pen_val,
                'lasso_weights': self.weights,
                'nuc': True
                }

    def _get_pen_val_max_from_pro(self, X, y):
        return nuclear_norm_max(X=X, y=y,
                                fit_intercept=self.fit_intercept,
                                weights=self.weights,
                                model_type=self._model_type)


class GlmNuclearNormCVPath(CVPathMixin, GlmCVSinglePen):
    @autoassign
    def __init__(self, weights=None, **kws):
        GlmCVSinglePen.__init__(self, **kws)

    def _get_base_est_params(self):
        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'weights': self.weights,
                }

    def _get_solve_path_kws(self):

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'lasso_weights': self.weights,
                'nuc': True,

                'lasso_pen_seq': self.pen_val_seq_
                }
