from ya_glm.models.linear_regression import LinRegMixin

from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, \
    GlmLassoENet, GlmLassoENetCVPath
from ya_glm.pen_glm.GroupLasso import GlmGroupLasso, GlmGroupLassoCVPath, \
    GlmGroupLassoENet, GlmGroupLassoENetCVPath
from ya_glm.pen_glm.Ridge import GlmRidge, GlmRidgeCVPath

from ya_glm.fcp.GlmFcp import GlmFcpFitLLA, GlmGroupFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpCV
from ya_glm.add_init_params import add_init_params

from ya_glm.lla.lla import solve_lla
from .glm_solver import solve_glm, solve_glm_path


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
    solve_glm = staticmethod(solve_glm)

    def _get_defualt_init(self):
        # return LassoCV()
        est = Lasso(fit_intercept=self.fit_intercept,
                    opt_kws=self.opt_kws,
                    standardize=self.standardize)

        return LassoCV(estimator=est)


class GroupFcpLLA(LinRegMixin, GlmGroupFcpFitLLA):
    solve_lla = staticmethod(solve_lla)
    solve_glm = staticmethod(solve_glm)

    def _get_defualt_init(self):
        est = GroupLasso(groups=self.groups,
                         fit_intercept=self.fit_intercept,
                         opt_kws=self.opt_kws,
                         standardize=self.standardize)

        return GroupLassoCV(estimator=est)

####################
# Cross-validation #
####################


class LassoCV(GlmLassoCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_init_params(GlmLassoCVPath)
    def __init__(self, estimator=Lasso()): pass


class LassoENetCV(GlmLassoENetCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_init_params(GlmLassoENetCVPath)
    def __init__(self, estimator=LassoENet()): pass


class GroupLassoCV(GlmGroupLassoCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_init_params(GlmLassoCVPath)
    # gruops=[] is hack to get around required positional arugment
    # TODO: is this how want to handle this issue?
    def __init__(self, estimator=GroupLasso(groups=[])): pass


class GroupLassoENetCV(GlmGroupLassoENetCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_init_params(GlmLassoENetCVPath)
    def __init__(self, estimator=GroupLassoENet(groups=[])): pass


class RidgeCV(GlmRidgeCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_init_params(GlmRidgeCVPath)
    def __init__(self, estimator=Ridge()): pass


class FcpLLACV(GlmFcpCV):

    @add_init_params(GlmFcpCV)
    def __init__(self, estimator=FcpLLA()): pass


class GroupFcpLLACV(GlmFcpCV):

    @add_init_params(GlmFcpCV)
    def __init__(self, estimator=GroupFcpLLA(groups=[])): pass
