from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath

from ya_glm.glm_pen_max_lasso import nuclear_norm_max
from ya_glm.processing import check_estimator_type


class GlmNuclearNorm(GlmLasso):

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """
        kws = super()._get_solve_kws()
        kws['nuc'] = True
        return kws

    def _get_pen_val_max_from_pro(self, X, y):

        loss_func, loss_kws = self.get_loss_info()

        return nuclear_norm_max(X=X, y=y,
                                fit_intercept=self.fit_intercept,
                                loss_func=loss_func,
                                loss_kws=loss_kws,
                                weights=self.weights)


class GlmNuclearNormCVPath(GlmLassoCVPath):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmNuclearNorm)
