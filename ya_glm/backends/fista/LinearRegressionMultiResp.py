from ya_glm.models.linear_regression_multi_resp import LinRegMultiResponseMixin

from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.NuclearNorm import GlmNuclearNorm, GlmNuclearNormCVPath
from ya_glm.pen_glm.MultiTaskLasso import GlmMultiTaskLasso, \
    GlmMultiTaskLassoCVPath,\
    GlmMultiTaskLassoENet, GlmMultiTaskLassoENetCVPath

from ya_glm.add_init_params import add_init_params

from .glm_solver import solve_glm, solve_glm_path


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
