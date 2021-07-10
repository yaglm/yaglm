import numpy as np
from copy import deepcopy
from textwrap import dedent

from ya_glm.utils import is_multi_response
from ya_glm.processing import process_weights_group_lasso
from ya_glm.opt.linear_regression import LinRegLoss, LinRegMultiRespLoss
from ya_glm.opt.logistic_regression import LogRegLoss
from ya_glm.opt.penalty import LassoPenalty, RidgePenalty, \
    WithIntercept, TikhonovPenalty
from ya_glm.opt.GroupLasso import GroupLasso
from ya_glm.opt.mat_penalty import RowLasso, NuclearNorm, \
    MatricizeEntrywisePen, \
    MatWithIntercept
from ya_glm.opt.utils import decat_coef_inter_vec, decat_coef_inter_mat
from ya_glm.opt.fista import solve_fista
from ya_glm.opt.base import Func, Sum

_solve_glm_params = dedent("""
X: array-like, shape (n_samples, n_features)
    The training covariate data.

y: array-like, shape (n_samples, )
    The training response data.

loss_func: str
    Which GLM loss function to use.

loss_kws: dict
    Keyword arguments for loss function.

fit_intercept: bool
    Whether or not to fit an intercept.

lasso_pen: None, float
    (Optional) The L1 penalty parameter value.

lasso_weights: None, array-like, shape (n_features, )
    (Optional) The L1 penalty feature weights.

groups: None, list of array-like
    (Optional) The group indicies for group Lasso.

L1to2: bool
    For matrix coefficients whether or not to use the L1 to L2 (i.e. sum of row L2 norms) norm or just the entrywise Lasso.

nuc: bool
    For matrix coefficients, whether or not to use nuclear norm.

ridge_pen: None, float
    (Optional) The L2 penalty parameter value.

ridge_weights: None, array-like, shape (n_features, )
    (Optional) The L2 penalty feature weights.

tikhonov: None, array-like shape (n_features, n_features)
    TODO
""")


_fista_params = dedent("""

coef_init: None, array-like, shape (n_features, )
    (Optional) Initialization for the coefficient.

intercept_init: None, float
    (Optional) Initialization for the intercept.

precomp_lip: None, float
    (Optional) Precomputed Lipschitz constant.

xtol: float, None
    The change in X stopping criterion. If provided, we terminate the algorithm if |x_new - x_current|_max < x_tol

rtol: None, float
    The relative tolerance criterion for the objective function. If provided, we terminate the algorithm if obj(x_current) - obj(x_new) < rtol * obj(x_current)

atol: None, float
    The absolute tolerance criterion for the objective function. If provided, we terminate the algorithm if obj(x_current) - obj(x_new) < obj(x_current)

max_iter: int
    The maximum number of optimiation steps to take.

tracking_level: int
    How much optmization data to track; larger values mean we track more data.
""")

_solve_glm_out = dedent("""
coef, intercept, opt_data

coef: array-like, shape (n_features, )
    The estimated coefficient.

intercept: None or float
    The estimated intercept -- if one was requested.

opt_data: dict
    The optmization history output.
""")


def solve_glm(X, y,
              loss_func='lin_reg',
              loss_kws={},
              fit_intercept=True,

              lasso_pen=None,
              lasso_weights=None,
              groups=None,
              L1to2=False,
              nuc=False,
              ridge_pen=None,
              ridge_weights=None,
              tikhonov=None,

              coef_init=None,
              intercept_init=None,
              precomp_lip=None,
              xtol=1e-4,
              rtol=None,
              atol=None,
              max_iter=1000,
              tracking_level=0):

    #######################
    # setup loss function #
    #######################

    loss_func = get_glm_loss(loss_func=loss_func, loss_kws=loss_kws,
                             X=X, y=y, fit_intercept=fit_intercept,
                             precomp_lip=precomp_lip)

    #############################
    # pre process penalty input #
    #############################

    # in case we passed a loss_func object, make sure fit_intercpet
    # agrees with loss_func
    fit_intercept = loss_func.fit_intercept

    if lasso_pen is None and lasso_weights is not None:
        lasso_pen = 1

    if ridge_pen is None and \
            (ridge_weights is not None or tikhonov is not None):
        ridge_pen = 1

    is_mr = is_multi_response(y)
    is_already_mat_pen = False

    # check
    if is_mr:
        # not currently supported for multi response
        assert groups is None
        assert tikhonov is None
    else:
        if nuc:
            raise ValueError("Nuclear norm not applicable"
                             " to vector coefficients")
    #################
    # Lasso penalty #
    #################
    if lasso_pen is None:
        lasso = None

    elif groups is not None:
        lasso_weights = process_weights_group_lasso(groups=groups,
                                                    weights=lasso_weights)

        lasso = GroupLasso(groups=groups,
                           mult=lasso_pen, weights=lasso_weights)

    elif is_mr and L1to2:
        lasso = RowLasso(mult=lasso_pen, weights=lasso_weights)
        is_already_mat_pen = True

    elif is_mr and nuc:
        lasso = NuclearNorm(mult=lasso_pen, weights=lasso_weights)
        is_already_mat_pen = True

    else:
        lasso = LassoPenalty(mult=lasso_pen, weights=lasso_weights)

    ##############
    # L2 penalty #
    ##############

    if ridge_pen is None:
        ridge = None

    elif tikhonov is not None:
        assert ridge_weights is None  # cant have both ridge_weights and tikhonov
        ridge = TikhonovPenalty(mult=ridge_pen, mat=tikhonov)

    else:
        ridge = RidgePenalty(mult=ridge_pen, weights=ridge_weights)

    # possibly format penalties for matrix coefficients
    if is_mr:
        if ridge is not None:
            ridge = MatricizeEntrywisePen(func=ridge)

        if lasso is not None and not is_already_mat_pen:
            lasso = MatricizeEntrywisePen(func=lasso)

    # possibly add intercept
    if fit_intercept:
        if ridge is not None:
            if is_mr:
                ridge = MatWithIntercept(func=ridge)
            else:
                ridge = WithIntercept(func=ridge)

        if lasso is not None:
            if is_mr:
                lasso = MatWithIntercept(func=lasso)
            else:
                lasso = WithIntercept(func=lasso)

    # if there is a ridge penalty add it to the loss funcion
    # TODO: do we want to do this for vanilla ridge + lasso?
    # or should we put lasso and ride together
    if ridge is not None:
        loss_func = Sum([loss_func, ridge])

    #####################
    # set initial value #
    #####################
    init_val = process_init(X, y, fit_intercept=fit_intercept,
                            coef_init=coef_init, intercept_init=intercept_init)

    ############################
    # solve problem with FISTA #
    ############################
    coef, out = solve_fista(smooth_func=loss_func,
                            init_val=init_val,
                            non_smooth_func=lasso,
                            step='lip',
                            accel=True,
                            restart=True,
                            max_iter=max_iter,
                            xtol=xtol,
                            rtol=rtol,
                            atol=atol,
                            tracking_level=tracking_level)

    # format output
    if fit_intercept:
        if is_mr:
            coef, intercept = decat_coef_inter_mat(coef)
        else:
            coef, intercept = decat_coef_inter_vec(coef)
    else:
        intercept = None

    return coef, intercept, out


solve_glm.__doc__ = dedent("""
Solve a penalized GLM problem using the FISTA algorithm.

Parameters
----------
{}

{}
Output
------
{}
""".format(_solve_glm_params, _fista_params, _solve_glm_out))


def solve_glm_path(X, y,
                   lasso_pen_seq=None, ridge_pen_seq=None,
                   loss_func='lin_reg', loss_kws={},
                   fit_intercept=True,
                   precomp_lip=None,
                   # generator=True,
                   check_decr=True,
                   **kws):
    """
    Fits a GLM along a tuning parameter path using the homotopy method.


    Parameters
    -----------
    all arguments are the same as solve_glm_fista except the following

    L1_pen_seq: None, array-like
        The L1 penalty parameter tuning sequence.

    ridge_pen_seq: None, array-like
        The L2 penalty parameter tuning sequence. If both L1_pen_seq and ridge_pen_seq are provided they should be the same length.

    check_decr: bool
        Whether or not to check the L1_pen_seq/ridge_pen_seq are monotonically decreasing.

    Output
    ------

    either a generator yielding
        fit_out: dict
        param_setting: dict

    or a list whith these entries.

    """
    # TODO: possibly re-write this so we can do tikhinov precomputation stuff once

    param_path = process_param_path(lasso_pen_seq=lasso_pen_seq,
                                    ridge_pen_seq=ridge_pen_seq,
                                    check_decr=check_decr)

    # out = []  # in case we want to return

    # this will precompute the lipschitz constant
    loss_func = get_glm_loss(loss_func=loss_func, loss_kws=loss_kws,
                             X=X, y=y, fit_intercept=fit_intercept,
                             precomp_lip=precomp_lip)

    # possibly get initializers
    if 'coef_init' in kws:
        coef = kws['coef_init']
        del kws['coef_init']
    else:
        coef = None

    if 'intercept_init' in kws:
        intercept = kws['intercept_init']
        del kws['intercept_init']
    else:
        intercept = None

    # fit each path value
    for params in param_path:

        # Solve this path element!
        coef, intercept, opt_data = solve_glm(X=X, y=y,
                                              loss_func=loss_func,
                                              coef_init=coef,
                                              intercept_init=intercept,
                                              **params,
                                              **kws)

        # format output
        fit_out = {'coef': coef,
                   'intercept': intercept,
                   'opt_data': opt_data}

        # if generator:
        yield fit_out, params

    #     else:
    #         out.append((fit_out, params))

    # return out


def get_glm_loss(X, y,
                 loss_func='lin_reg', loss_kws={},
                 fit_intercept=True, precomp_lip=None):
    """
    Returns an GLM loss function object.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, )
        The training response data.

    fit_intercept: bool
        Whether or not to fit an intercept.

    loss_func: str
        Which GLM loss function to use.
        Must be one of ['linear_regression', 'logistic_regression'].
        This may also be an instance of ya_glm.opt.base.Func.

    precomp_lip: None, float
        (Optional) Precomputed Lipchitz constant

    Output
    ------
    glm_loss: ya_glm.opt.Func
        The GLM loss function object.
    """

    if isinstance(loss_func, Func):
        return loss_func

    assert loss_func in ['lin_reg',
                         'lin_reg_mr',
                         'log_reg']

    if loss_func == 'lin_reg':
        obj_class = LinRegLoss

    elif loss_func == 'lin_reg_mr':
        obj_class = LinRegMultiRespLoss

    elif loss_func == 'log_reg':
        obj_class = LogRegLoss

    return obj_class(X=X, y=y, fit_intercept=fit_intercept, lip=precomp_lip,
                     **loss_kws)


def process_init(X, y, fit_intercept=True, coef_init=None, intercept_init=None):
    """
    Processes the initializer.

    Parameters
    ----------

    Outout
    ------
    init_val: array-like
        The initial value. Shape is (n_features, ), (n_features + 1, )
        (n_features, n_responses) or (n_features, n_responses + 1)
    """

    is_mr = is_multi_response(y)

    # initialize coefficient
    if coef_init is None:
        if is_mr:
            coef_shape = (X.shape[1], y.shape[1])
        else:
            coef_shape = (X.shape[1],)
        coef_init = np.zeros(coef_shape)

    # initialize intercept
    if intercept_init is None:
        if is_mr:
            intercept_init = np.zeros(y.shape[1])
        else:
            intercept_init = 0

    # maybe concatenate
    if fit_intercept:
        if is_mr:
            init_val = np.vstack([intercept_init, coef_init])
        else:
            init_val = np.concatenate([[intercept_init], coef_init])
    else:
        init_val = deepcopy(coef_init)

    return init_val


def process_param_path(lasso_pen_seq=None, ridge_pen_seq=None, check_decr=True):

    if check_decr:
        if lasso_pen_seq is not None:
            assert all(np.diff(lasso_pen_seq) <= 0)

        if ridge_pen_seq is not None:
            assert all(np.diff(ridge_pen_seq) <= 0)

    if lasso_pen_seq is not None and ridge_pen_seq is not None:
        assert len(lasso_pen_seq) == len(ridge_pen_seq)

        param_path = [{'lasso_pen': lasso_pen_seq[i],
                       'ridge_pen': ridge_pen_seq[i]}
                      for i in range(len(lasso_pen_seq))]

    elif lasso_pen_seq is not None:
        param_path = [{'lasso_pen': lasso_pen_seq[i]}
                      for i in range(len(lasso_pen_seq))]

    elif ridge_pen_seq is not None:
        param_path = [{'ridge_pen': ridge_pen_seq[i]}
                      for i in range(len(ridge_pen_seq))]

    else:
        raise ValueError("One of lasso_pen_seq, ridge_pen_seq should be provided ")

    return param_path
