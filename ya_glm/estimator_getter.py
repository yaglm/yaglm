# loss funcs
from ya_glm.models.linear_regression import LinRegMixin
from ya_glm.models.multinomial import MultinomialMixin

from ya_glm.models.logistic_regression import LogRegMixin
from ya_glm.models.linear_regression_multi_resp import \
    LinRegMultiResponseMixin
from ya_glm.models.huber_regression import HuberRegMixin, \
    HuberRegMultiResponseMixin
from ya_glm.models.poisson_regression import PoissonRegMixin,\
    PoissonRegMultiResponseMixin
from ya_glm.models.quantile_regression import QuantileRegMixin

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


# andersoncd solvers
from ya_glm.backends.andersoncd.glm_solver import solve_glm as \
    solve_glm_andersoncd
from ya_glm.backends.andersoncd.glm_solver import solve_glm_path \
    as solve_glm_path_andersoncd

# cvxpy solvers
from ya_glm.backends.cvxpy.glm_solver import solve_glm as solve_glm_cvxpy
from ya_glm.backends.cvxpy.glm_solver import solve_glm_path \
    as solve_glm_path_cvxpy


# other
from ya_glm.add_init_params import add_init_params
from ya_glm.info import _MULTI_RESP_LOSSES, _MULTI_RESP_PENS, _CONCAVEABLE_PENS


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

    elif loss_func == 'multinomial':
        return MultinomialMixin

    elif loss_func == 'poisson':
        return PoissonRegMixin

    elif loss_func == 'poisson_mr':
        return PoissonRegMultiResponseMixin

    elif loss_func == 'quantile':
        return QuantileRegMixin

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


def get_solver(backend='fista'):
    """
    Parameters
    ----------
    backend: str, dict

    """

    # get solver
    if type(backend) == str:
        if backend == 'fista':
            solve_glm = solve_glm_fista
            solve_glm_path = solve_glm_path_fista


        elif backend == 'andersoncd':
            solve_glm = solve_glm_andersoncd
            solve_glm_path = solve_glm_path_andersoncd

        elif backend == 'cvxpy':
            solve_glm = solve_glm_cvxpy
            solve_glm_path = solve_glm_path_cvxpy

    else:
        solve_glm = backend.get('solve_glm', None)
        solve_glm_path = backend.get('solve_glm_path', None)

    solve_glm = staticmethod(solve_glm)
    if solve_glm_path is not None:
        solve_glm_path = staticmethod(solve_glm_path)

    return solve_glm, solve_glm_path

# TODO: handle static method + None


def get_pen_glm(loss_func='linear_regression',
                penalty='lasso',
                backend='fista'):

    if penalty in _MULTI_RESP_PENS:
        assert loss_func in _MULTI_RESP_LOSSES

    MODEL_MIXIN = get_model_mixin(loss_func=loss_func)
    GLM, GLM_CV = get_penalty(penalty=penalty)
    solve_glm, solve_glm_path = get_solver(backend=backend)

    # TODO-HACK: for reasons I do not understand I needed to
    # do this to get Estimator() to work below
    temp = {}
    temp['solve_glm'] = solve_glm
    temp['solve_glm_path'] = solve_glm_path

    # setup estimator #
    ###################

    class Estimator(MODEL_MIXIN, GLM):
        # solve_glm = solve_glm
        solve_glm = temp['solve_glm']

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
            # solve_glm_path = solve_glm_path
            solve_glm_path = temp['solve_glm_path']

            @add_init_params(GLM_CV)
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
    solve_glm = get_solver(backend=backend)[0]

    temp = {'solve_glm': solve_glm}  # TODO-HACK: see above

    # get default initializer
    Default, DefaultCV = get_pen_glm(loss_func=loss_func,
                                     penalty=penalty,
                                     backend=backend)

    ###################
    # setup estimator #
    ###################

    class Estimator(MODEL_MIXIN, GLM_FCP):
        solve_lla = staticmethod(solve_lla)
        # solve_glm = solve_glm
        solve_glm = temp['solve_glm']

        @add_init_params(GLM_FCP, MODEL_MIXIN)
        def __init__(self): pass

    if 'group' in penalty:

        def _get_defualt_init(self):
            # return DefaultCV()
            est = Default(groups=self.groups,
                          fit_intercept=self.fit_intercept,
                          opt_kws=self.opt_kws,
                          standardize=self.standardize)

            return DefaultCV(estimator=est)

        Estimator._get_defualt_init = _get_defualt_init
        estimator = Estimator(groups=[])

    else:

        def _get_defualt_init(self):
            # return DefaultCV()
            est = Default(fit_intercept=self.fit_intercept,
                          opt_kws=self.opt_kws,
                          standardize=self.standardize)
            return DefaultCV(estimator=est)

        Estimator._get_defualt_init = _get_defualt_init
        estimator = Estimator()

    class EstimatorCV(GLM_FCP_CV):

        @add_init_params(GLM_FCP_CV)
        def __init__(self, estimator=estimator): pass

    ####################################
    # setup cross-validation estimator #
    ####################################

    return Estimator, EstimatorCV
