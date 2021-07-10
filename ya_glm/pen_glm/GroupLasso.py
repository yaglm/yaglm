from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, GlmLassoENet, \
    GlmLassoENetCVPath

from ya_glm.add_init_params import add_init_params
from ya_glm.glm_pen_max_lasso import group_lasso_max
from ya_glm.processing import check_estimator_type


class GlmGroupLasso(GlmLasso):

    @add_init_params(GlmLasso)
    def __init__(self, groups, weights='size'): pass

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """
        kws = super()._get_solve_kws()
        kws['groups'] = self.groups
        return kws

    def _get_pen_val_max_from_pro(self, X, y):

        loss_func, loss_kws = self.get_loss_info()

        return group_lasso_max(X=X, y=y,
                               groups=self.groups,
                               fit_intercept=self.fit_intercept,
                               loss_func=loss_func,
                               loss_kws=loss_kws,
                               weights=self.weights)


class GlmGroupLassoCVPath(GlmLassoCVPath):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmGroupLasso)


class GlmGroupLassoENet(GlmLassoENet):

    @add_init_params(GlmLassoENet)
    def __init__(self, groups, lasso_weights='size'): pass

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """
        kws = super()._get_solve_kws()
        kws['groups'] = self.groups
        return kws

    def _get_pen_val_max_from_pro(self, X, y):
        loss_func, loss_kws = self.get_loss_info()

        l1_max = group_lasso_max(X=X, y=y,
                                 groups=self.groups,
                                 fit_intercept=self.fit_intercept,
                                 loss_func=loss_func,
                                 loss_kws=loss_kws,
                                 weights=self.lasso_weights)

        return l1_max / self.l1_ratio


class GlmGroupLassoENetCVPath(GlmLassoENetCVPath):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmGroupLassoENet)
