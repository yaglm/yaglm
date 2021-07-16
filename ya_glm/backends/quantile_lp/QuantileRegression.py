from ya_glm.glm_loss.quantile_regression import QuantileRegMixin

from ya_glm.pen_glms.GlmVanilla import GlmVanilla

from ya_glm.pen_glms.GlmRidge import GlmRidge, GlmRidgeCVPath

from ya_glm.pen_glms.GlmLasso import GlmLasso, GlmLassoCVPath, GlmENet, \
    GlmENetCVPath

from ya_glm.pen_glms.GlmAdaptiveLasso import GlmAdaptiveLasso,\
    GlmAdaptiveLassoCVPath, GlmAdaptiveENet, GlmAdaptiveENetCVPath

from ya_glm.pen_glms.GlmFcpLLA import GlmFcpLLA, GlmFcpLLACV

from ya_glm.init_signature import add_from_classes

from .glm_solver import solve_glm, solve_glm_path


##############
# Single fit #
##############


class Vanilla(QuantileRegMixin, GlmVanilla):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmVanilla, QuantileRegMixin)
    def __init__(self): pass


class Ridge(QuantileRegMixin, GlmRidge):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmRidge, QuantileRegMixin)
    def __init__(self): pass


class Lasso(QuantileRegMixin, GlmLasso):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmLasso, QuantileRegMixin)
    def __init__(self): pass


class ENet(QuantileRegMixin, GlmENet):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmENet, QuantileRegMixin)
    def __init__(self): pass


class AdaptiveLasso(QuantileRegMixin, GlmAdaptiveLasso):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmAdaptiveLasso, QuantileRegMixin)
    def __init__(self): pass

    def _get_defualt_init(self):
        est = Lasso(**self._kws_for_default_init())
        return LassoCV(estimator=est)


class AdaptiveENet(QuantileRegMixin, GlmAdaptiveENet):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmAdaptiveENet, QuantileRegMixin)
    def __init__(self): pass

    def _get_defualt_init(self):
        est = ENet(**self._kws_for_default_init())
        return ENetCV(estimator=est)


class FcpLLA(QuantileRegMixin, GlmFcpLLA):
    solve_glm = staticmethod(solve_glm)

    @add_from_classes(GlmFcpLLA, QuantileRegMixin)
    def __init__(self): pass

    def _get_defualt_init(self):
        est = Lasso(**self._kws_for_default_init())
        return LassoCV(estimator=est)

####################
# Cross-validation #
####################


class RidgeCV(GlmRidgeCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_from_classes(GlmRidgeCVPath)
    def __init__(self, estimator=Ridge()): pass

# TODO: do we really want to use the path algorithm here that calls cvxpy?
class LassoCV(GlmLassoCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_from_classes(GlmLassoCVPath)
    def __init__(self, estimator=Lasso()): pass


class ENetCV(GlmENetCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_from_classes(GlmENetCVPath)
    def __init__(self, estimator=ENet()): pass


class AdpativeLassoCV(GlmAdaptiveLassoCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_from_classes(GlmAdaptiveLassoCVPath)
    def __init__(self, estimator=AdaptiveLasso()): pass


class AdaptiveENetCV(GlmAdaptiveENetCVPath):
    solve_glm_path = staticmethod(solve_glm_path)

    @add_from_classes(GlmAdaptiveENetCVPath)
    def __init__(self, estimator=AdaptiveENet()): pass


class FcpLLACV(GlmFcpLLACV):

    @add_from_classes(GlmFcpLLACV)
    def __init__(self, estimator=FcpLLA()): pass
