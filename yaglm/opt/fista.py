import numpy as np
from copy import deepcopy
from time import time

from yaglm.opt.base import Zero
from yaglm.opt.stopping import check_decreasing_loss, check_no_change


def solve_fista(smooth_func, init_val, non_smooth_func=None,
                step=1,
                accel=True,
                restart=True,
                backtracking=False,
                max_iter=200,
                xtol=1e-5,
                rtol=None,
                atol=None,
                bt_max_steps=20,
                bt_shrink=0.5,
                bt_grow=1.1,
                tracking_level=0):

    """
    Solve an optimzation problem in the form of

    min_x smoot_func(x) + non_smooth_func(x)

    using ISTA/FISTA.

    Parameters
    ----------
    smooth_func: pyunlocbox.functions.func
        The smooth part of the loss function. This object should implement smooth_func.grad() and smooth_func.eval().

    init_val: array-like
        The value to initialize from.

    non_smooth_func: None, pyunlocbox.functions.func
        The (optional) non-smooth part of the loss function. This object should implement non_smooth_func.prox() and non_smooth_func.eval().

    step: float, 'lip'
        The step size. This is either the constant step size or the base step size if backtracking is used.

    accel: bool
        Whether or not to use FISTA acceleration.

    restart: bool
        Whether or not to restart the acceleration scheme. See (13) from https://bodono.github.io/publications/adap_restart.pdf
        for the strategy we employ.

    backtracking: bool
        Whether or not to do a backtracking line search (e.g. if the Lipschtiz constant is not known).

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

    Output
    ------
    value: array-like
        The solution.

    opt_data: dict
        Additional optimization data e.g. loss history, etc.
    """

    start_time = time()

    # if there is no non_smooth function then this is just the zero function
    if non_smooth_func is None:
        non_smooth_func = Zero()

    # Setup update stesps
    def eval_obj(x):
        return smooth_func.eval(x) + non_smooth_func.eval(x)

    def prox_grad_update(x, step):
        return non_smooth_func.prox(x - step * smooth_func.grad(x), step)

    # TODO: we could save some computation by inputing the precomputed computed non_smooth_func.eval(x)
    def Q(x, y, step):
        diff = x - y
        diff_sq = (diff ** 2).sum()
        return smooth_func.eval(y) + \
            (x - y).ravel().T @ smooth_func.grad(y).ravel() +\
            (0.5 / step) * diff_sq + \
            non_smooth_func.eval(x)

    def backtracking_search(x, step, bt_iter_prev):
        # increase the step size if the last one was accepted
        if bt_iter_prev == 0 and bt_grow is not None:
            step *= bt_grow  # they do this in copt

        for bt_iter in range(bt_max_steps):
            x_new = prox_grad_update(x, step)

            obj_new = eval_obj(x_new)

            if obj_new <= Q(x_new, x, step):
                break
            else:
                step *= bt_shrink

        return x_new, step, bt_iter

    # setup values
    value = np.array(init_val)
    value_prev = value.copy()
    if accel:
        value_aux = value.copy()
        value_aux_prev = value.copy()
        t, t_prev = 1, 1

    else:
        value_aux, value_aux_prev, t, t_prev = None, None, None, None
    bt_iter = 0

    if (atol or rtol) and tracking_level == 0:
        # if we are using atol or rtol then we need to compute
        # the objective function anyways
        tracking_level = 1

    # how much optimization history should we track
    history = {}
    if tracking_level >= 1:
        history['objective'] = [eval_obj(value)]

        if backtracking:
            history['bt_iter'] = []
            history['step'] = []

    if tracking_level >= 2:
        history['diff_norm'] = []

    if restart:
        history['restarts'] = []

    # set learning rate from Lipchitz constant
    if step == 'lip':
        glip = smooth_func.grad_lip
        assert glip is not None
        step = 1 / glip

    x_stop = False
    obj_stop = False
    bt_iter = 0
    for it in range(int(max_iter)):

        ###############
        # Update step #
        ###############
        if accel:
            # with acceleration

            if backtracking:
                # FISTA with backtracking
                value_aux, step, bt_iter = \
                    backtracking_search(value, step, bt_iter)
            else:
                # FISTA with constant step
                value_aux = prox_grad_update(value, step)

            # FISTA step
            t = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            value = value_aux + \
                ((t_prev - 1) / t) * (value_aux - value_aux_prev)

            if restart:
                # see equation (13) of TODO
                if (value_prev - value_aux).ravel().T @ \
                        (value_aux - value_aux_prev).ravel() > 0:
                    t, t_prev = 1, 1
                    history['restarts'].append(it)

        elif backtracking:
            # Backtracking line search
            value, step, bt_iter = backtracking_search(value, step, bt_iter)

        else:
            # Constant step size
            value = prox_grad_update(value, step)

        # possibly track data
        if tracking_level >= 1:
            history['objective'].append(eval_obj(value))

            if backtracking:
                history['bt_iter'].append(bt_iter)
                history['step'].append(step)

        # check X difference stopping triterion
        x_stop, diff_norm = check_no_change(current=value,
                                            prev=value_prev,
                                            tol=xtol,
                                            norm='max')

        if tracking_level >= 2:
            history['diff_norm'].append(diff_norm)

        # objective function criteria
        if atol is not None or rtol is not None:
            obj_stop = \
                check_decreasing_loss(current_loss=history['objective'][-1],
                                      prev_loss=history['objective'][-2],
                                      abs_tol=atol, rel_tol=rtol,
                                      on_increase='ignore')

        # maybe stop
        if x_stop or obj_stop:
            break
        else:
            value_prev = value.copy()
            if accel:
                value_aux_prev = value_aux
                t_prev = deepcopy(t)

    out = {'runtime': time() - start_time,
           'history': history,
           'x_stop': x_stop,
           'obj_stop': obj_stop,
           'iter': it}

    return value, out
