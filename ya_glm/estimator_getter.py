# loss funcs
from ya_glm.models.linear_regression import LinRegMixin
from ya_glm.models.logistic_regression import LogRegMixin
from ya_glm.models.linear_regression_multi_resp import \
    LinRegMultiResponseMixin
from ya_glm.models.huber_regression import HuberRegMixin, \
    HuberRegMultiResponseMixin

# penalties
from ya_glm.pen_glm.Vanilla import GlmVanilla
from ya_glm.pen_glm.Lasso import GlmLasso, GlmLassoCVPath, \
    GlmLassoENet, GlmLassoENetCVPath
from ya_glm.pen_glm.GroupLasso import GlmGroupLasso, GlmGroupLassoCVPath, \
    GlmGroupLassoENet, GlmGroupLassoENetCVPath
from ya_glm.pen_glm.Ridge import GlmRidge, GlmRidgeCVPath
from ya_glm.pen_glm.MultiTaskLasso import GlmMultiTaskLasso, GlmMultiTaskLassoCVPath, \
    GlmMultiTaskLassoENet, GlmMultiTaskLassoENetCVPath
from ya_glm.pen_glm.NuclearNorm import GlmNuclearNorm, GlmNuclearNormCVPath

# concave
from ya_glm.fcp.GlmFcp import GlmFcpFitLLA, GlmMultiTaskFcpFitLLA, \
    GlmNuclearNormFcpFitLLA, GlmGroupFcpFitLLA
from ya_glm.fcp.GlmFcpCV import GlmFcpCV
from ya_glm.lla.lla import solve_lla

# fista solvers
from ya_glm.backends.fista.glm_solver import solve_glm as solve_glm_fista
from ya_glm.backends.fista.glm_solver import solve_glm_path \
    as solve_glm_path_fista
from ya_glm.backends.fista.fcp_lla_solver import WL1SolverGlm \
    as WL1SolverGlm_fista

# andersoncd solvers
from ya_glm.backends.andersoncd.glm_solver import solve_glm as \
    solve_glm_andersoncd
from ya_glm.backends.andersoncd.glm_solver import solve_glm_path \
    as solve_glm_path_andersoncd
from ya_glm.backends.andersoncd.fcp_lla_solver import WL1SolverGlm \
    as WL1SolverGlm_andersoncd

# other
from ya_glm.add_init_params import add_init_params


_MULTI_RESP_LOSSES = ['lin_reg_mr', 'huber_reg_mr']
_MULTI_RESP_PENS = ['multi_task_lasso', 'multi_task_lasso_enet',
                    'nuclear_norm']
_CONCAVEABLE_PENS = ['lasso', 'group_lasso', 'multi_task_lasso',
                     'nuclear_norm']


def get_model_mixin(loss_func='lin_reg'):
    # loss function
    if loss_func == 'lin_reg':
        return LinRegMixin

    elif loss_func == 'lin_reg_mr':
        return LinRegMultiResponseMixin

    if loss_func == 'huber_reg':
        return HuberRegMixin

    elif loss_func == 'huber_reg_mr':
        return HuberRegMultiResponseMixin

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

    elif penalty == 'multi_task_lasso':
        return GlmMultiTaskLasso, GlmMultiTaskLassoCVPath

    elif penalty == 'multi_task_lasso_enet':
        return GlmMultiTaskLassoENet, GlmMultiTaskLassoENetCVPath

    elif penalty == 'nuclear_norm':
        return GlmNuclearNorm, GlmNuclearNormCVPath

    else:
        raise ValueError("Bad input for penalty: {}".format(penalty))


def get_fcp_penalty(penalty='lasso'):
    # Valid FCP models
    assert penalty in _CONCAVEABLE_PENS

    # penalty
    if penalty == 'lasso':
        return GlmFcpFitLLA, GlmFcpCV

    elif penalty == 'group_lasso':
        return GlmGroupFcpFitLLA, GlmFcpCV,

    elif penalty == 'multi_task_lasso':
        return GlmMultiTaskFcpFitLLA, GlmFcpCV

    elif penalty == 'nuclear_norm':
        return GlmNuclearNormFcpFitLLA, GlmFcpCV

    else:
        raise ValueError("Bad input for penalty: {}".format(penalty))


def get_pen_glm(loss_func='linear_regression',
                penalty='lasso',
                backend='fista'):
    
    if penalty in _MULTI_RESP_PENS:
        assert loss_func in _MULTI_RESP_LOSSES

    MODEL_MIXIN = get_model_mixin(loss_func=loss_func)
    GLM, GLM_CV = get_penalty(penalty=penalty)

    # get solver
    if type(backend) == str and backend == 'fista':
        solve_glm_impl = solve_glm_fista
        solve_glm_path_impl = solve_glm_path_fista

    elif type(backend) == str and backend == 'andersoncd':
        solve_glm_impl = solve_glm_andersoncd
        solve_glm_path_impl = solve_glm_path_andersoncd

    else:
        solve_glm_impl = backend.get('solve_glm', None)
        solve_glm_path_impl = backend.get('solve_glm_path', None)

    ###################
    # setup estimator #
    ###################

    class Estimator(MODEL_MIXIN, GLM):
        solve = staticmethod(solve_glm_impl)

        @add_init_params(GLM, MODEL_MIXIN)
        def __init__(self): pass

    ####################################
    # setup cross-validation estimator #
    ####################################
    if GLM_CV is not None:

        if 'group' in penalty:
            # TODO-HACK: this avoids issue of required positional argument
            # is this how we want to handle this issue?
            estimator = Estimator(groups=[])
        else:
            estimator = Estimator()

        class EstimatorCV(GLM_CV):
            solve_path = staticmethod(solve_glm_path_impl)

            @add_init_params(GlmLassoENetCVPath)
            def __init__(self, estimator=estimator): pass

    else:
        EstimatorCV = None

    return Estimator, EstimatorCV


# TODO: add in other penalty types eg group, multi task, nuc
def get_fcp_glm(loss_func='linear_regression', penalty='lasso',
                backend='fista'):

    # get base model class
    MODEL_MIXIN = get_model_mixin(loss_func=loss_func)
    GLM_FCP, GLM_FCP_CV = get_fcp_penalty(penalty=penalty)

    # get wl1 solver
    if type(backend) == str and backend == 'fista':
        WL1_impl = WL1SolverGlm_fista

    elif type(backend) == str and backend == 'andersoncd':
        WL1_impl = WL1SolverGlm_andersoncd

    else:
        WL1_impl = backend.get('wl1', None)

    # get default initializer
    Default, DefaultCV = get_pen_glm(loss_func=loss_func,
                                     penalty=penalty,
                                     backend=backend)

    ###################
    # setup estimator #
    ###################

    if 'group' in penalty:
        class Estimator(MODEL_MIXIN, GLM_FCP):
            solve_lla = staticmethod(solve_lla)
            base_wl1_solver = WL1_impl

            @add_init_params(MODEL_MIXIN)
            def __init__(self): pass

            def _get_defualt_init(self):
                # return DefaultCV()
                est = Default(groups=self.groups,
                              fit_intercept=self.fit_intercept,
                              opt_kws=self.opt_kws,
                              standardize=self.standardize)

                return DefaultCV(estimator=est)

        estimator = Estimator(groups=[])

    else:
        class Estimator(MODEL_MIXIN, GLM_FCP):
            solve_lla = staticmethod(solve_lla)
            base_wl1_solver = WL1_impl

            @add_init_params(MODEL_MIXIN)
            def __init__(self): pass

            def _get_defualt_init(self):
                # return DefaultCV()
                est = Default(fit_intercept=self.fit_intercept,
                              opt_kws=self.opt_kws,
                              standardize=self.standardize)
                return DefaultCV(estimator=est)

        estimator = Estimator()

    class EstimatorCV(GLM_FCP_CV):

        @add_init_params(GlmFcpCV)
        def __init__(self, estimator=estimator): pass

    ####################################
    # setup cross-validation estimator #
    ####################################

    return Estimator, EstimatorCV
