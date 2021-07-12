import numpy as np
from copy import deepcopy
from time import time

from textwrap import dedent

from ya_glm.opt.stopping import check_decreasing_loss, check_no_change
from ya_glm.lla.utils import safe_concat

# TODO: add option for unpenalized variable
# TODO: add option for initializing lasso from previous solution


_lla_docs = dict(
    opt_options=dedent("""
    n_steps: int
        Number of LLA steps to take.

    xtol: float, None
        The change in x tolerance stopping criterion based on the L_infy norm.

    atol: float, None
        Absolute tolerance for loss based stopping criterion.

    rtol: float, None
        Relative tolerance for loss based stopping criterion.

    tracking_level: int
        How much optimization data to store at each step. Lower values means less informationed is stored.

    verbosity: int
        How much information to print out. Lower values means less print out.
        """),


    opt_prob=dedent("""
    wl1_solver:
        TODO

    penalty_fcn: fclsp.penalty.FoldedPenalty
        The penalty function.

    init: array-like
        The value at which to initalize the LLA algorithm.

    init_upv: None, array-like
        The value at which to initialize the (optional) unpenalized variable.

    transform: callable
        Transforms the penalized variable into the object whom we apply the concave penalty to.

    """),

    out=dedent("""
    solution: array-like
        The solution of the penalized variable.

    solution_upv: None, array-like
        The solution of the unpenalized variable.

    opt_data: dict
        Data tracked during the optimization procedure e.g. the loss function.
    """),

    refs=dedent("""
    Fan, J., Xue, L. and Zou, H., 2014. Strong oracle optimality of folded concave penalized estimation. Annals of statistics, 42(3), p.819.

        """)
)


def solve_lla(wl1_solver, penalty_fcn, init,
              init_upv=None, transform=abs,
              n_steps=1, xtol=1e-4, atol=None, rtol=None,
              tracking_level=1, verbosity=0):

    def evaluate_obj(x, upv, T):
        # compute current objective function
        base_loss = wl1_solver.loss(value=x, upv=upv)

        T = transform(x)
        pen_loss = penalty_fcn.eval(T)

        obj = base_loss + pen_loss
        return obj, base_loss, pen_loss

    ######################
    # format initializer #
    ######################

    current = deepcopy(init)
    current_upv = deepcopy(init_upv)
    T = transform(current)

    if xtol is not None:
        if current_upv is not None:
            prev = deepcopy(safe_concat(current, current_upv))
        else:
            prev = deepcopy(current)

    ##############################
    # optimization data tracking #
    ##############################
    opt_data = {}

    if (atol is not None or rtol is not None) and tracking_level == 0:
        tracking_level = 1

    if tracking_level >= 1:
        # opt_data['init'] = deepcopy(init)
        obj, base_loss, pen_loss = evaluate_obj(current, current_upv, T=T)

        opt_data['base_loss'] = [base_loss]  # the base loss fucntion
        opt_data['pen_loss'] = [pen_loss]  # penaltyloss
        opt_data['obj'] = [obj]  # loss + penalty

    if tracking_level >= 2:
        if xtol is not None:
            opt_data['diff_norm'] = []  # difference between successive iterates

    #################
    # Run algorithm #
    #################

    start_time = time()

    step = 0  # in case n_steps = 0
    x_stop = False
    obj_stop = False
    for step in range(int(n_steps)):

        if verbosity >= 1:
            print("Step {}, {:1.2f} seconds after start".
                  format(step + 1, time() - start_time))

        ###############
        # Make update #
        ###############

        # majorize concave penalty
        # T has already been computed as T = transform(current)
        L1_weights = penalty_fcn.grad(T)

        # solve weighted Lasso problem
        current, current_upv, other_data = \
            wl1_solver.solve(L1_weights=L1_weights,
                             opt_init=current,
                             opt_init_upv=current_upv)

        ############################################
        # check stopping conditions and track data #
        ############################################

        T = None  # tells us to compute T below

        # track objective function data
        if tracking_level >= 1:
            T = transform(current)

            obj, base_loss, pen_loss = evaluate_obj(current, current_upv, T)

            opt_data['obj'].append(obj)
            opt_data['base_loss'].append(base_loss)
            opt_data['pen_loss'].append(pen_loss)

        # x change criteria
        if xtol is not None:
            if current_upv is not None:
                _current = safe_concat(current, current_upv)
            else:
                _current = current

            x_stop, diff_norm = check_no_change(current=_current,
                                                prev=prev,
                                                tol=xtol,
                                                norm='max')

            if tracking_level >= 2:
                opt_data['diff_norm'].append(diff_norm)

        if tracking_level >= 1:
            # objective function change criterion
            # if the objective has stopped decreasing we are done!
            obj_stop = check_decreasing_loss(current_loss=opt_data['obj'][-1],
                                             prev_loss=opt_data['obj'][-2],
                                             abs_tol=atol, rel_tol=rtol,
                                             on_increase='ignore')

        # maybe stop the algorithm
        if x_stop or obj_stop:
            break
        elif xtol is not None:

            # set prev for next interation
            if current_upv is not None:
                prev = deepcopy(safe_concat(current, current_upv))
            else:
                prev = deepcopy(current)

            # compute transform for next iteration if it has not already
            # been computed
            if T is None:
                T = transform(current)

    opt_data['runtime'] = time() - start_time
    opt_data['final_step'] = step
    opt_data['obj_stop'] = obj_stop
    opt_data['x_stop'] = x_stop

    return current, current_upv, opt_data
