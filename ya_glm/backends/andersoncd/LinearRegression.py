from ya_glm.models.linear_regression import LinRegMixin

from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath

from ya_glm.fcp.GlmFcp import GlmFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpCV
from ya_glm.lla.lla import solve_lla
from ya_glm.add_init_params import add_init_params


from .glm_solver import solve_glm, solve_glm_path
from .fcp_lla_solver import WL1SolverGlm


##############
# Single fit #
##############


class Vanilla(LinRegMixin, GlmVanilla):
    solve = staticmethod(solve_glm)


class Lasso(LinRegMixin, GlmLasso):
    solve = staticmethod(solve_glm)


class FcpLLA(LinRegMixin, GlmFcpFitLLA):
    solve_lla = staticmethod(solve_lla)
    base_wl1_solver = WL1SolverGlm

    def _get_defualt_init(self):
        return LassoCV()


####################
# Cross-validation #
####################


class LassoCV(GlmLassoCVPath):
    solve_path = staticmethod(solve_glm_path)

    @add_init_params(GlmLassoCVPath)
    def __init__(self, estimator=Lasso()): pass


class FcpLLACV(GlmFcpCV):

    @add_init_params(GlmFcpCV)
    def __init__(self, estimator=FcpLLA()): pass
