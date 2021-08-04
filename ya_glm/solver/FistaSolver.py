from copy import deepcopy

from ya_glm.GlmSolver import GlmSolver
from ya_glm.ConstraintConfig import ConstraintConfig
from ya_glm.opt.penalty.concave_penalty import get_penalty_func
from ya_glm.opt.penalty.vec import LassoPenalty, RidgePenalty, \
    WithIntercept, TikhonovPenalty
from ya_glm.opt.penalty.GroupLasso import GroupLasso
from ya_glm.opt.penalty.mat_penalty import MultiTaskLasso, NuclearNorm, \
    MatricizeEntrywisePen, MatWithIntercept
from ya_glm.opt.penalty.composite_structured import CompositeGroup,\
    CompositeMultiTaskLasso, CompositeNuclearNorm
from ya_glm.opt.penalty.constraints import PositiveOrthant

from ya_glm.opt.utils import decat_coef_inter_vec, decat_coef_inter_mat
from ya_glm.opt.fista import solve_fista
from ya_glm.opt.base import Sum, Func
from ya_glm.opt.glm_loss.get import get_glm_loss, _LOSS_FUNC_CLS2STR

from ya_glm.solver.utils import process_param_path
from ya_glm.autoassign import autoassign


class FistaSolver(GlmSolver):
    """
    Solves a penalized GLM problem using the FISTA algorithm.

    Parameters
    ----------
    max_iter: int
        Maximum number of iterations.

    xtol: float, None
        Stopping criterion based on max norm of successive iteration differences i.e. stop if max(x_new - x_prev) < xtol.

    rtol: float, None
        Stopping criterion based on the relative difference of successive loss function values i.e. stop if abs(loss_new - loss_prev)/loss_new < rtol.

    atol: float, None
        Stopping criterion based on the absolute difference of successive loss function values i.e. stop if abs(loss_new - loss_prev) < atol.

    bt_max_steps: int
        Maximum number of backtracking steps to take.

    bt_shrink: float
        How much to shrink the step size in each backtracking step. Should lie strictly in the unit interval.

    bt_grow: float, None
        (Optional) How much to grow the step size each iteraction when using backgracking.

    tracking_level: int
        How much data to track.
    """

    @autoassign
    def __init__(self,
                 max_iter=1000,
                 xtol=1e-4,
                 rtol=None,
                 atol=None,
                 bt_max_steps=20,
                 bt_shrink=0.5,
                 bt_grow=1.1,
                 tracking_level=0): pass

    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def _get_avail_losses(self):
        return ['lin_reg', 'huber',
                'log_reg', 'multinomial',
                'poisson']  # no quantile

    # TODO: compute lipchitz constant,
    # possibly do stuff with tikhonov
    def setup(self, X, y, loss, penalty, sample_weight=None):
        pass

    def solve(self, X, y, loss, penalty,
              fit_intercept=True,
              sample_weight=None,
              coef_init=None,
              intercept_init=None
              ):
        """
        Solves a penalized GLM problem. See docs for ya_glm.GlmSolver.
        """

        if loss.name not in self._get_avail_losses():
            raise ValueError("{} loss not available; this solver only"
                             "implements {}".format(loss.name,
                                                    self._get_avail_losses()))

        if isinstance(penalty, ConstraintConfig):
            solve = solve_constrained_glm
        else:
            solve = solve_glm

        return solve(X=X, y=y,
                     loss=loss,
                     fit_intercept=fit_intercept,
                     sample_weight=sample_weight,
                     coef_init=coef_init,
                     intercept_init=intercept_init,
                     **penalty.get_solve_kws(),
                     **self.get_solve_kws())

    def solve_path(self, X, y, loss, penalty_seq,
                   fit_intercept=True,
                   sample_weight=None,
                   coef_init=None,
                   intercept_init=None):
        """
        Solves a sequence of penalized GLM problem using a path algorithm. See docs for ya_glm.GlmSolver.
        """

        if loss.name not in self._get_avail_losses():
            raise ValueError("{} loss not available; this solver only"
                             "implements {}".format(loss.name,
                                                    self._get_avail_losses()))

        if coef_init is not None or intercept_init is not None:
            # TODO do we want to allow this? perhaps warn?
            raise NotImplementedError

        return solve_glm_path(X=X, y=y,
                              loss=loss,
                              fit_intercept=fit_intercept,
                              sample_weight=sample_weight,
                              **penalty_seq.get_solve_kws(),
                              **self.get_solve_kws())

    def has_path_algo(self, loss, penalty):
        """
        Yes this solver has an available path algorithm!
        """
        return True


def solve_glm(X, y,
              loss,
              fit_intercept=True,
              sample_weight=None,

              lasso_pen_val=None,
              lasso_weights=None,
              groups=None,
              multi_task=False,
              nuc=False,
              ridge_pen_val=None,
              ridge_weights=None,
              tikhonov=None,

              nonconvex_func=None,
              nonconvex_func_kws=None,

              coef_init=None,
              intercept_init=None,
              xtol=1e-4,
              rtol=None,
              atol=None,
              max_iter=1000,
              bt_max_steps=20,
              bt_shrink=0.5,
              bt_grow=1.1,
              tracking_level=0):

    """
    Sets up and solves a penalized GLM problem using fista.
    For documentation of the GLM arguments see TODO.


    For documentaion of the optimization arguments see ya_glm.opt.fista.solve_fista.
    """

    #######################
    # setup loss function #
    #######################

    is_mr = y.ndim > 1 and y.shape[1] > 1

    if not isinstance(loss, Func):
        # get loss function object
        loss_func = get_glm_loss(X=X, y=y, loss=loss,
                                 fit_intercept=fit_intercept,
                                 sample_weight=sample_weight)

        # precompute Lipchitz constant
        loss_func.setup()
    else:
        loss_func = loss

    if _LOSS_FUNC_CLS2STR[type(loss_func)] == 'quantile':
        raise NotImplementedError("fista solver does not support quantile loss")

    # in case we passed a loss_func object, make sure fit_intercpet
    # agrees with loss_func
    fit_intercept = loss_func.fit_intercept

    #####################
    # set initial value #
    #####################
    if coef_init is None or intercept_init is None:
        init_val = loss_func.default_init()
    else:
        init_val = loss_func.cat_intercept_coef(intercept_init, coef_init)

    #############################
    # pre process penalty input #
    #############################

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
    if lasso_pen_val is None:
        lasso = None

    elif nonconvex_func is None:

        if groups is not None:

            lasso = GroupLasso(groups=groups,
                               mult=lasso_pen_val, weights=lasso_weights)

        elif is_mr and multi_task:
            lasso = MultiTaskLasso(mult=lasso_pen_val, weights=lasso_weights)
            is_already_mat_pen = True

        elif is_mr and nuc:
            lasso = NuclearNorm(mult=lasso_pen_val, weights=lasso_weights)
            is_already_mat_pen = True

        else:
            lasso = LassoPenalty(mult=lasso_pen_val, weights=lasso_weights)

    else:

        # set penaly for compoisite function
        func = get_penalty_func(pen_func=nonconvex_func,
                                pen_val=lasso_pen_val,
                                pen_func_kws=nonconvex_func_kws)

        if groups is not None:
            lasso = CompositeGroup(groups=groups, func=func)

        elif is_mr and multi_task:
            lasso = CompositeMultiTaskLasso(func=func)
            is_already_mat_pen = True

        elif is_mr and nuc:
            lasso = CompositeNuclearNorm(func=func)
            is_already_mat_pen = True

        else:
            lasso = func

    ##############
    # L2 penalty #
    ##############

    if ridge_pen_val is None:
        ridge = None

    elif tikhonov is not None:
        assert ridge_weights is None  # cant have both ridge_weights and tikhonov
        ridge = TikhonovPenalty(mult=ridge_pen_val, mat=tikhonov)

    else:
        ridge = RidgePenalty(mult=ridge_pen_val, weights=ridge_weights)

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

    # setup step size/backtracking
    if loss_func.grad_lip is not None:
        # use Lipchtiz constant if it is available
        step = 'lip'
        backtracking = False
    else:
        step = 1  # TODO: perhaps smarter base step size?
        backtracking = True

    ############################
    # solve problem with FISTA #
    ############################
    coef, out = solve_fista(smooth_func=loss_func,
                            init_val=init_val,
                            non_smooth_func=lasso,
                            step=step,
                            backtracking=backtracking,
                            accel=True,
                            restart=True,
                            max_iter=max_iter,
                            xtol=xtol,
                            rtol=rtol,
                            atol=atol,
                            bt_max_steps=bt_max_steps,
                            bt_shrink=bt_shrink,
                            bt_grow=bt_grow,
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


def solve_glm_path(X, y, loss,
                   lasso_pen_seq=None, ridge_pen_seq=None,
                   fit_intercept=True,
                   sample_weight=None,
                   # generator=True,
                   check_decr=True,
                   **kws):
    """
    Fits a GLM along a tuning parameter path using the homotopy method.
    Each subproblem is solved using FISTA.

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

    # get the GLm loss object
    loss_func = get_glm_loss(X=X, y=y, loss=loss,
                             fit_intercept=fit_intercept,
                             sample_weight=sample_weight)

    # precompute Lipchitz constant
    loss_func.setup()

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
                                              loss=loss_func,
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


def solve_constrained_glm(X, y,
                          loss,
                          fit_intercept=True,
                          sample_weight=None,

                          pos=False,

                          coef_init=None,
                          intercept_init=None,
                          xtol=1e-4,
                          rtol=None,
                          atol=None,
                          max_iter=1000,
                          bt_max_steps=20,
                          bt_shrink=0.5,
                          bt_grow=1.1,
                          tracking_level=0):

    """
    Sets up and solves a constrained GLM problem using fista.
    For documentation of the GLM arguments see TODO.

    For documentaion of the optimization arguments see ya_glm.opt.fista.solve_fista.
    """

    #######################
    # setup loss function #
    #######################

    is_mr = y.ndim > 1 and y.shape[1] > 1

    loss_func = get_glm_loss(X=X, y=y, loss=loss,
                             fit_intercept=fit_intercept,
                             sample_weight=sample_weight)

    if _LOSS_FUNC_CLS2STR[type(loss_func)] == 'quantile':
        raise NotImplementedError("fista solver does not support quantile loss")

    ####################
    # setup constraint #
    ####################

    if pos:
        constraint = PositiveOrthant()

    else:
        constraint = None

    # make sure we dont constrain the intercept
    if constraint is not None and fit_intercept:
        if is_mr:
            constraint = MatWithIntercept(func=constraint)
        else:
            constraint = WithIntercept(func=constraint)

    #####################
    # set initial value #
    #####################
    if coef_init is None or intercept_init is None:
        init_val = loss_func.default_init()
    else:
        init_val = loss_func.cat_intercept_coef(intercept_init, coef_init)

    ##############
    # L2 penalty #
    ##############

    # setup step size/backtracking
    if loss_func.grad_lip is not None:
        # use Lipchtiz constant if it is available
        step = 'lip'
        backtracking = False
    else:
        step = 1  # TODO: perhaps smarter base step size?
        backtracking = True

    coef, out = solve_fista(smooth_func=loss_func,
                            init_val=init_val,
                            non_smooth_func=constraint,
                            step=step,
                            backtracking=backtracking,
                            accel=True,
                            restart=True,
                            max_iter=max_iter,
                            xtol=xtol,
                            rtol=rtol,
                            atol=atol,
                            bt_max_steps=bt_max_steps,
                            bt_shrink=bt_shrink,
                            bt_grow=bt_grow,
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
