from ya_glm.models.linear_regression_multi_resp import LinRegMultiResponseMixin

from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.NuclearNorm import GlmNuclearNorm, GlmNuclearNormCVPath
from ya_glm.pen_glm.MultiTaskLasso import GlmMultiTaskLasso, \
    GlmMultiTaskLassoCVPath,\
    GlmMultiTaskLassoENet, GlmMultiTaskLassoENetCVPath
from ya_glm.fcp.GlmFcp import GlmMultiTaskFcpFitLLA, \
    GlmNuclearNormFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpCV

from ya_glm.add_init_params import add_init_params

from ya_glm.lla.lla import solve_lla
from .glm_solver import solve_glm, solve_glm_path
from .fcp_lla_solver import WL1SolverGlm


##############
# Single fit #
##############


class Vanilla(LinRegMultiResponseMixin, GlmVanilla):
    solve = staticmethod(solve_glm)


class MultiTaskLasso(LinRegMultiResponseMixin, GlmMultiTaskLasso):
    solve = staticmethod(solve_glm)


class MultiTaskLassoENet(LinRegMultiResponseMixin, GlmMultiTaskLassoENet):
    solve = staticmethod(solve_glm)


class NuclearNorm(LinRegMultiResponseMixin, GlmNuclearNorm):
    solve = staticmethod(solve_glm)


class MultiTaskFcpLLA(LinRegMultiResponseMixin, GlmMultiTaskFcpFitLLA):
    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = WL1SolverGlm

    def _get_defualt_init(self):
        est = MultiTaskLasso(fit_intercept=self.fit_intercept,
                             opt_kws=self.opt_kws,
                             standardize=self.standardize)

        return MultiTaskLassoCV(estimator=est)


class NuclearNormFcpLLA(LinRegMultiResponseMixin, GlmNuclearNormFcpFitLLA):
    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = WL1SolverGlm

    def _get_defualt_init(self):
        est = NuclearNorm(fit_intercept=self.fit_intercept,
                          opt_kws=self.opt_kws,
                          standardize=self.standardize)

        return NuclearNormCV(estimator=est)


####################
# Cross-validation #
####################


class MultiTaskLassoCV(GlmMultiTaskLassoCVPath):
    solve_path = staticmethod(solve_glm_path)

    @add_init_params(GlmMultiTaskLassoCVPath)
    def __init__(self, estimator=MultiTaskLasso()): pass


class MultiTaskLassoENetCV(GlmMultiTaskLassoENetCVPath):
    solve_path = staticmethod(solve_glm_path)

    @add_init_params(GlmMultiTaskLassoENetCVPath)
    def __init__(self, estimator=MultiTaskLassoENet()): pass


class NuclearNormCV(GlmNuclearNormCVPath):
    solve_path = staticmethod(solve_glm_path)

    @add_init_params(GlmNuclearNormCVPath)
    def __init__(self, estimator=NuclearNorm()): pass


class MultiTaskFcpLLACV(GlmFcpCV):

    @add_init_params(GlmFcpCV)
    def __init__(self, estimator=MultiTaskFcpLLA()): pass


class NuclearNormFcpLLACV(GlmFcpCV):

    @add_init_params(GlmFcpCV)
    def __init__(self, estimator=NuclearNormFcpLLA()): pass
