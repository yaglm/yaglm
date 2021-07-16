# loss funcs
from ya_glm.glm_loss.linear_regression import LinRegMixin
from ya_glm.glm_loss.multinomial import MultinomialMixin

from ya_glm.glm_loss.logistic_regression import LogRegMixin
from ya_glm.glm_loss.linear_regression_multi_resp import \
    LinRegMultiResponseMixin
from ya_glm.glm_loss.huber_regression import HuberRegMixin, \
    HuberRegMultiResponseMixin
from ya_glm.glm_loss.poisson_regression import PoissonRegMixin,\
    PoissonRegMultiResponseMixin
from ya_glm.glm_loss.quantile_regression import QuantileRegMixin

# penalties
from ya_glm.pen_glms.GlmVanilla import GlmVanilla
from ya_glm.pen_glms.GlmRidge import GlmRidge, \
    GlmRidgeCVPath, GlmRidgeCVGridSearch
from ya_glm.pen_glms.GlmLasso import GlmLasso, GlmENet, \
    GlmLassoCVPath, GlmLassoCVGridSearch, \
    GlmENetCVPath, GlmENetCVGridSearch

from ya_glm.pen_glms.GlmAdaptiveLasso import \
    GlmAdaptiveLasso, GlmAdaptiveENet, \
    GlmAdaptiveLassoCVPath, GlmAdaptiveLassoCVGridSearch, \
    GlmAdaptiveENetCVPath, GlmAdaptiveENetCVGridSearch

from ya_glm.pen_glms.GlmFcpLLA import GlmFcpLLA, GlmFcpLLACV


# fista solvers
from ya_glm.backends.fista.glm_solver import solve_glm as solve_glm_fista
from ya_glm.backends.fista.glm_solver import solve_glm_path \
    as solve_glm_path_fista
from ya_glm.backends.fista.WL1SolverGlm import WL1SolverGlm as WL1SolverGlmFista


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
from ya_glm.init_signature import add_from_classes, add_multi_resp_params
from ya_glm.info import _MULTI_RESP_LOSSES, _MULTI_RESP_PENS
from ya_glm.lla.WeightedLassoSolver import WL1SolverGlm


def get_loss_mixin(loss_func='lin_reg'):

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


def get_penalty(penalty='lasso', has_path_algo=True):
    """
    Parameters
    ----------
    penalty: str
        Name of penalty

    has_path_algo: bool
        If the GLM has a path algorithm.

    Output
    ------
    Est, EstCV
    """

    # penalty
    if penalty == 'vanilla':
        Est, EstCV = GlmVanilla, None

    elif penalty == 'ridge':
        Est = GlmRidge
        if has_path_algo:
            EstCV = GlmRidgeCVPath
        else:
            EstCV = GlmRidgeCVGridSearch

    elif penalty == 'lasso':
        Est = GlmLasso

        if has_path_algo:
            EstCV = GlmLassoCVPath
        else:
            EstCV = GlmLassoCVGridSearch

    elif penalty == 'enet':
        Est = GlmENet

        if has_path_algo:
            EstCV = GlmENetCVPath
        else:
            EstCV = GlmENetCVGridSearch

    elif penalty == 'adpt_lasso':
        Est = GlmAdaptiveLasso

        if has_path_algo:
            EstCV = GlmAdaptiveLassoCVPath
        else:
            EstCV = GlmAdaptiveLassoCVGridSearch

    elif penalty == 'adpt_enet':
        Est = GlmAdaptiveENet

        if has_path_algo:
            EstCV = GlmAdaptiveENetCVPath
        else:
            EstCV = GlmAdaptiveENetCVGridSearch

    elif penalty == 'fcp_lla':
        Est = GlmFcpLLA
        EstCV = GlmFcpLLACV

    else:
        raise ValueError("Bad input for penalty: {}".format(penalty))

    return Est, EstCV


def get_solver(backend='fista'):
    """
    Parameters
    ----------
    backend: str, dict

    Output
    ------
    solve_glm, solve_glm_path, WL1Solver

    solve_glm: callable
        A function that solves a single penalized GLM problem.

    solve_glm_path: None, callable
        (Optional) A function that solves a penalized GLM path.

    WL1Solver: class
        An weighted L1 solver class for the LLA algorithm.
    """

    wl1_solver = WL1SolverGlm

    # get solver
    if type(backend) == str:
        if backend == 'fista':
            solve_glm = solve_glm_fista
            solve_glm_path = solve_glm_path_fista
            wl1_solver = WL1SolverGlmFista

        elif backend == 'andersoncd':
            solve_glm = solve_glm_andersoncd
            solve_glm_path = solve_glm_path_andersoncd

        elif backend == 'cvxpy':
            solve_glm = solve_glm_cvxpy
            solve_glm_path = solve_glm_path_cvxpy

    else:
        solve_glm = backend.get('solve_glm', None)
        solve_glm_path = backend.get('solve_glm_path', None)

    if solve_glm is None:
        raise ValueError("No solver found")

    solve_glm = staticmethod(solve_glm)
    if solve_glm_path is not None:
        solve_glm_path = staticmethod(solve_glm_path)

    return solve_glm, solve_glm_path, wl1_solver


def get_pen_glm(loss_func='lin_reg',
                penalty='lasso',
                backend='fista'):

    if penalty in _MULTI_RESP_PENS:
        assert loss_func in _MULTI_RESP_LOSSES

    solve_glm, solve_glm_path, WL1Solver = get_solver(backend=backend)
    LOSS_MIXIN = get_loss_mixin(loss_func=loss_func)
    GLM, GLM_CV = get_penalty(penalty=penalty,
                              has_path_algo=solve_glm_path is not None)

    if penalty == 'fcp_lla':
        solve_glm_path = None

    # TODO-HACK: for reasons I do not understand I needed to
    # do this to get Estimator() to work below
    temp = {'solve_glm': solve_glm,
            'solve_glm_path': solve_glm_path,
            'WL1Solver': WL1Solver}

    ##############################################
    # setup defaults inits for concave penalties #
    ##############################################

    if penalty in ['fcp_lla', 'adpt_lasso']:
        Default, DefaultCV = get_pen_glm(loss_func=loss_func,
                                         penalty='lasso',
                                         backend=backend)

    elif penalty == 'adpt_enet':
        Default, DefaultCV = get_pen_glm(loss_func=loss_func,
                                         penalty='enet',
                                         backend=backend)

    ###################
    # setup estimator #
    ###################

    if penalty == 'fcp_lla':

        class Estimator(LOSS_MIXIN, GLM):
            solve_glm = temp['solve_glm']  # TODO-HACK: see above
            WL1Solver = temp['WL1Solver']  # TODO-HACK: see above

            @add_multi_resp_params(add=LOSS_MIXIN.is_multi_resp)
            @add_from_classes(GLM, LOSS_MIXIN)
            def __init__(self): pass

            # TODO: make safe
            def _get_defualt_init(self):
                est = Default(**self._kws_for_default_init(c=Default))
                return DefaultCV(estimator=est)

    elif penalty in ['adpt_lasso', 'adpt_enet']:

        class Estimator(LOSS_MIXIN, GLM):
            solve_glm = temp['solve_glm']  # TODO-HACK: see above

            @add_multi_resp_params(add=LOSS_MIXIN.is_multi_resp)
            @add_from_classes(GLM, LOSS_MIXIN)
            def __init__(self): pass

            # TODO: make safe
            def _get_defualt_init(self):
                est = Default(**self._kws_for_default_init(c=Default))
                return DefaultCV(estimator=est)

    else:

        class Estimator(LOSS_MIXIN, GLM):
            solve_glm = temp['solve_glm']  # TODO-HACK: see above

            @add_multi_resp_params(add=LOSS_MIXIN.is_multi_resp)
            @add_from_classes(GLM, LOSS_MIXIN)
            def __init__(self): pass

    ####################################
    # setup cross-validation estimator #
    ####################################
    if GLM_CV is not None:

        class EstimatorCV(GLM_CV):
            # solve_glm_path = solve_glm_path
            solve_glm_path = temp['solve_glm_path']  # TODO-HACK: see above

            @add_from_classes(GLM_CV)
            def __init__(self, estimator=Estimator()): pass

    else:
        EstimatorCV = None

    return Estimator, EstimatorCV
