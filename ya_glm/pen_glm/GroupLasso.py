import numpy as np

from ya_glm.autoassign import autoassign
from ya_glm.utils import get_lasso_and_L2_from_enet
from ya_glm.glm_pen_max_lasso import group_lasso_max

from ya_glm.Glm import Glm
from ya_glm.GlmCV import GlmCVSinglePen, GlmCVENet
from ya_glm.ENetCVPath import ENetCVPathMixin
from ya_glm.cv.CVPath import CVPathMixin


class GlmGroupLasso(Glm):

    @autoassign
    def __init__(self, groups, pen_val=1, weights='size', **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):

        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'groups': self.groups,
                'lasso_pen': self.pen_val,
                'lasso_weights': weights,
                }

    def _get_pen_val_max_from_pro(self, X, y):
        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        return group_lasso_max(X=X, y=y,
                               fit_intercept=self.fit_intercept,
                               weights=weights,
                               groups=self.groups,
                               model_type=self._model_type)


class GlmGroupLassoCVPath(CVPathMixin, GlmCVSinglePen):
    @autoassign
    def __init__(self, groups, weights='size', **kws):
        GlmCVSinglePen.__init__(self, **kws)

    def _get_base_est_params(self):
        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'groups': self.groups,
                'weights': self.weights,
                }

    def _get_solve_path_kws(self):

        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'groups': self.groups,
                'lasso_weights': weights,

                'lasso_pen_seq': self.pen_val_seq_
                }


class GlmGroupLassoENet(Glm):

    @autoassign
    def __init__(self, groups, pen_val=1, l1_ratio=0.5, weights='size',
                 tikhonov=None, **kws):
        super().__init__(**kws)

    def _get_solve_glm_kws(self):
        lasso_pen, L2_pen = get_lasso_and_L2_from_enet(pen_val=self.pen_val,
                                                       l1_ratio=self.l1_ratio)

        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'groups': self.groups,
                'lasso_pen': lasso_pen,
                'lasso_weights': weights,

                'L2_pen': L2_pen,
                'tikhonov': self.tikhonov
                }

    def _get_pen_val_max_from_pro(self, X, y):

        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        l1_max = group_lasso_max(X=X, y=y,
                                 fit_intercept=self.fit_intercept,
                                 weights=weights,
                                 groups=self.groups,
                                 model_type=self._model_type)
        return l1_max / self.l1_ratio


class GlmGroupLassoENetCVPath(ENetCVPathMixin, GlmCVENet):

    @autoassign
    def __init__(self, groups, weights='size',
                 tikhonov=None, **kws):
        GlmCVENet.__init__(self, **kws)

    def _get_extra_base_params(self):

        return {'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws,
                'standardize': self.standardize,

                'groups': self.groups,
                'weights': self.weights,
                'tikhonov': self.tikhonov
                }

    def _get_solve_path_base_kws(self):

        weights = process_weights_group_lasso(groups=self.groups,
                                              weights=self.weights)

        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws,

                'groups': self.groups,
                'lasso_weights': weights,
                'tikhonov': self.tikhonov
                }


def process_weights_group_lasso(groups, weights=None):
    """

    Parameters
    ----------
    groups: list of array-like

    weights: str or array-like
    """
    if weights is None:
        return None

    if weights == 'size':
        group_sizes = np.array([len(grp) for grp in groups])
        weights = 1 / np.sqrt(group_sizes)
    else:
        weights = weights
