from ya_glm.autoassign import autoassign
from ya_glm.utils import get_lasso_and_L2_from_enet
from ya_glm.glm_pen_max_lasso import get_L1toL2_max

from ya_glm.Glm import Glm
from ya_glm.GlmCV import GlmCVSinglePen, GlmCVENet
from ya_glm.ENetCVPath import ENetCVPathMixin
from ya_glm.cv.CVPath import CVPathMixin


class GlmRowLasso(Glm):

    @autoassign
    def __init__(self, pen_val=1, weights=None, **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):
        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'lasso_pen': self.pen_val,
                'lasso_weights': self.weights,
                'L1to2': True
                }

    def _get_pen_val_max_from_pro(self, X, y):
        return get_L1toL2_max(X=X, y=y,
                              fit_intercept=self.fit_intercept,
                              weights=self.weights,
                              model_type=self._model_type)


class GlmRowLassoCVPath(CVPathMixin, GlmCVSinglePen):
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
                'L1to2': True,

                'lasso_pen_seq': self.pen_val_seq_
                }


class GlmRowLassoENet(Glm):

    @autoassign
    def __init__(self, pen_val=1, l1_ratio=0.5, tikhonov=None, **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):
        lasso_pen, L2_pen = get_lasso_and_L2_from_enet(pen_val=self.pen_val,
                                                       l1_ratio=self.l1_ratio)

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'lasso_pen': lasso_pen,
                'L2_pen': L2_pen,
                'tikhonov': self.tikhonov,
                'L1to2': True
                }

    def _get_pen_val_max_from_pro(self, X, y):

        l1_max = get_L1toL2_max(X=X, y=y,
                                fit_intercept=self.fit_intercept,
                                weights=self.weights,
                                model_type=self._model_type)

        return l1_max / self.l1_ratio


class GlmRowLassoENetCVPath(ENetCVPathMixin, GlmCVENet):

    @autoassign
    def __init__(self, tikhonov=None, **kws):
        GlmCVENet.__init__(self, **kws)

    def _get_extra_base_params(self):

        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'tikhonov': self.tikhonov
                }

    def _get_solve_path_base_kws(self):

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'tikhonov': self.tikhonov,
                'L1to2': True
                }
