from ya_glm.models.linear_regression import LinRegMixin
from ya_glm.models.logistic_regression import LogRegMixin

from ya_glm.models.linear_regression_multi_resp import \
    LinRegMultiResponseMixin


# penalties
from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, \
    GlmLassoENet, GlmLassoENetCVPath
from ya_glm.pen_glm.GroupLasso import GlmGroupLasso, GlmGroupLassoCVPath, \
    GlmGroupLassoENet, GlmGroupLassoENetCVPath
from ya_glm.pen_glm.Ridge import GlmRidge, GlmRidgeCVPath
from ya_glm.pen_glm.RowLasso import GlmRowLasso, GlmRowLassoCVPath, \
    GlmRowLassoENet, GlmRowLassoENetCVPath
from ya_glm.pen_glm.NuclearNorm import GlmNuclearNorm, GlmNuclearNormCVPath

# concave
from ya_glm.fcp.GlmFcp import GlmFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpFitLLACV
from ya_glm.lla.lla import solve_lla

# solvers
from ya_glm.backends.fista.glm_solver import solve_glm as solve_glm_fista
from ya_glm.backends.fista.glm_solver import solve_glm_path \
    as solve_glm_path_fista
from ya_glm.backends.fista.fcp_lla_solver import WL1SolverGlm \
    as WL1SolverGlm_fista

from ya_glm.backends.celer.glm_solver import solve_glm as solve_glm_celer
from ya_glm.backends.celer.glm_solver import solve_glm_path \
    as solve_glm_path_celer
from ya_glm.backends.celer.fcp_lla_solver import WL1SolverGlm \
    as WL1SolverGlm_celer


def get_model_mixin(loss_func='lin_reg'):
    # loss function
    if loss_func == 'lin_reg':
        return LinRegMixin

    elif loss_func == 'lin_reg_mr':
        return LinRegMultiResponseMixin

    elif loss_func == 'log_reg':
        return LogRegMixin

    else:
        raise NotImplementedError("{} not supported".format(loss_func))


def get_penalty(penalty='lasso'):

    # penalty
    if penalty == 'vanilla':
        return GlmVanilla, None

    elif penalty == 'lasso':
        return GlmLasso, GlmLassoCVPath

    elif penalty == 'lasso_enet':
        return GlmLassoENet, GlmLassoENetCVPath

    elif penalty == 'group_lasso':
        return GlmGroupLasso, GlmGroupLassoCVPath,

    elif penalty == 'group_lasso_enet':
        return GlmGroupLassoENet, GlmGroupLassoENetCVPath

    elif penalty == 'ridge':
        return GlmRidge, GlmRidgeCVPath

    elif penalty == 'row_lasso':
        return GlmRowLasso, GlmRowLassoCVPath

    elif penalty == 'row_lasso_enet':
        return GlmRowLassoENet, GlmRowLassoENetCVPath

    elif penalty == 'nuclear_norm':
        return GlmNuclearNorm, GlmNuclearNormCVPath


def get_pen_glm(loss_func='linear_regression',
                penalty='lasso',
                backend='fista'):
    
    MODEL_MIXIN = get_model_mixin(loss_func=loss_func)
    GLM, GLM_CV = get_penalty(penalty=penalty)

    # get solver
    if type(backend) == str and backend == 'fista':
        solve_glm_impl = solve_glm_fista
        solve_glm_path_impl = solve_glm_path_fista

    elif type(backend) == str and backend == 'celer':
        solve_glm_impl = solve_glm_celer
        solve_glm_path_impl = solve_glm_path_celer

    else:
        solve_glm_impl = backend.get('solve_glm', None)
        solve_glm_path_impl = backend.get('solve_glm_path', None)

    # setup estimator
    class Estimator(MODEL_MIXIN, GLM):
        solve_glm = staticmethod(solve_glm_impl)

    # setup cross-validation
    if GLM_CV is not None:
        class EstimatorCV(GLM_CV):
            solve_path = staticmethod(solve_glm_path_impl)

            def _get_base_class(self):
                return Estimator

    else:
        EstimatorCV = None

    return Estimator, EstimatorCV


def get_fcp_model(loss_func='linear_regression', backend='fista'):

    # get base model class
    MODEL_MIXIN = get_model_mixin(loss_func=loss_func)

    # get wl1 solver
    if type(backend) == str and backend == 'fista':
        WL1_impl = WL1SolverGlm_fista

    elif type(backend) == str and backend == 'celer':
        WL1_impl = WL1SolverGlm_celer

    else:
        WL1_impl = backend.get('wl1', None)

    # get default initializer
    DefaultCV = get_pen_glm(loss_func=loss_func,
                            penalty='lasso',
                            backend=backend)[1]

    # setup model
    class Estimator(MODEL_MIXIN, GlmFcpFitLLA):
        solve_lla = staticmethod(solve_lla)
        base_wl1_solver = WL1_impl

        def _get_defualt_init(self):
            return DefaultCV(fit_intercept=self.fit_intercept,
                             opt_kws=self.opt_kws)

    # setup CV estimator
    class EstimatorCV(GlmFcpFitLLACV):
        def _get_base_class(self):
            return Estimator

    return Estimator, EstimatorCV
