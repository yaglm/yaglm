from ya_glm.models.logistic_regression import LogRegMixin

from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, \
    GlmLassoENet, GlmLassoENetCVPath
from ya_glm.pen_glm.GroupLasso import GlmGroupLasso, GlmGroupLassoCVPath, \
    GlmGroupLassoENet, GlmGroupLassoENetCVPath
from ya_glm.pen_glm.Ridge import GlmRidge, GlmRidgeCVPath

from ya_glm.fcp.GlmFcp import GlmFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpFitLLACV
from ya_glm.lla.lla import solve_lla

from .glm_solver import solve_glm, solve_glm_path
from .fcp_lla_solver import WL1SolverGlm


##############
# Single fit #
##############


class Vanilla(LogRegMixin, GlmVanilla):
    solve_glm = staticmethod(solve_glm)


class Lasso(LogRegMixin, GlmLasso):
    solve_glm = staticmethod(solve_glm)


class LassoENet(LogRegMixin, GlmLassoENet):
    solve_glm = staticmethod(solve_glm)


class GroupLasso(LogRegMixin, GlmGroupLasso):
    solve_glm = staticmethod(solve_glm)


class GroupLassoENet(LogRegMixin, GlmGroupLassoENet):
    solve_glm = staticmethod(solve_glm)


class Ridge(LogRegMixin, GlmRidge):
    solve_glm = staticmethod(solve_glm)


class FcpLLA(LogRegMixin, GlmFcpFitLLA):
    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = WL1SolverGlm

    def _get_defualt_init(self):
        return LassoCV(fit_intercept=self.fit_intercept,
                       opt_kws=self.opt_kws)


####################
# Cross-validation #
####################


class LassoCV(GlmLassoCVPath):
    solve_path = staticmethod(solve_glm_path)

    def _get_base_class(self):
        return Lasso


class LassoENetCV(GlmLassoENetCVPath):
    solve_path = staticmethod(solve_glm_path)

    def _get_base_class(self):
        return LassoENet


class GroupLassoCV(GlmGroupLassoCVPath):
    solve_path = staticmethod(solve_glm_path)

    def _get_base_class(self):
        return GroupLasso


class GroupLassoENetCV(GlmGroupLassoENetCVPath):
    solve_path = staticmethod(solve_glm_path)

    def _get_base_class(self):
        return GroupLassoENet


class RidgeCV(GlmRidgeCVPath):
    solve_path = staticmethod(solve_glm_path)

    def _get_base_class(self):
        return Ridge


class FcpLLACV(GlmFcpFitLLACV):

    def _get_base_class(self):
        return FcpLLA
