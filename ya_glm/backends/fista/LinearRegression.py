from ya_glm.models.linear_regression import LinRegMixin

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


class Vanilla(LinRegMixin, GlmVanilla):
    solve_glm = staticmethod(solve_glm)


class Lasso(LinRegMixin, GlmLasso):
    solve_glm = staticmethod(solve_glm)


class LassoENet(LinRegMixin, GlmLassoENet):
    solve_glm = staticmethod(solve_glm)


class GroupLasso(LinRegMixin, GlmGroupLasso):
    solve_glm = staticmethod(solve_glm)


class GroupLassoENet(LinRegMixin, GlmGroupLassoENet):
    solve_glm = staticmethod(solve_glm)


class Ridge(LinRegMixin, GlmRidge):
    solve_glm = staticmethod(solve_glm)


class FcpLLA(LinRegMixin, GlmFcpFitLLA):
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
