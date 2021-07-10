from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, GlmLassoENet, \
    GlmLassoENetCVPath

from ya_glm.glm_pen_max_lasso import get_L1toL2_max
from ya_glm.processing import check_estimator_type


class GlmMultiTaskLasso(GlmLasso):

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """
        kws = super()._get_solve_kws()
        kws['L1to2'] = True
        return kws

    def _get_pen_val_max_from_pro(self, X, y):

        loss_func, loss_kws = self.get_loss_info()

        return get_L1toL2_max(X=X, y=y,
                              fit_intercept=self.fit_intercept,
                              loss_func=loss_func,
                              loss_kws=loss_kws,
                              weights=self.weights)


class GlmMultiTaskLassoCVPath(GlmLassoCVPath):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmMultiTaskLasso)


class GlmMultiTaskLassoENet(GlmLassoENet):

    def _get_pen_val_max_from_pro(self, X, y):
        loss_func, loss_kws = self.get_loss_info()

        l1_max = get_L1toL2_max(X=X, y=y,
                                fit_intercept=self.fit_intercept,
                                loss_func=loss_func,
                                loss_kws=loss_kws,
                                weights=self.lasso_weights)

        return l1_max / self.l1_ratio


class GlmMultiTaskLassoENetCVPath(GlmLassoENetCVPath):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmMultiTaskLassoENet)
