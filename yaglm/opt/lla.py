from copy import deepcopy
from time import time

from yaglm.opt.stopping import check_decreasing_loss, check_no_change
from yaglm.opt.utils import safe_concat


def solve_lla(sub_prob, penalty_func,
              init, init_upv=None,
              sp_init=None, sp_upv_init=None, sp_other_data=None,
              transform=abs, objective=None,
              max_steps=1,
              tol=1e-5, rel_crit=False, stop_crit='x_max',
              tracking_level=1, verbosity=0):
    """
    Runs the local linear approximation algorithm. We only need the concave penalty function and a subroutine that solves the weighted Lasso-like subproblems.

    Parameters
    ----------
    sub_prob:
        An object that solves the weighted subproblems.

    penalty_func: fclsp.penalty.FoldedPenalty
        The penalty function applied to the possibly transformed values.

    init: array-like
        The value at which to initalize the LLA algorithm.

    init_upv: None, array-like
        The value at which to initialize the (optional) unpenalized variable.

    sp_init: None, array-like
        (Optional) Value at which to initialize the first weighted subproblem.

    sp_upv_init: None, array-like
        (Optional) Value at which to initialize the first weighted subproblem for the unpenalized variable.

    sp_other_data: None
        (Optional) Value at which to initialize the other subproblem data.

    transform: callable
        Transforms the penalized variable into the object whom we apply the concave penalty to.

    objective: None, callable(value, upv) -> float
        (Optinoal) Evaluates the full objective function.

    max_steps: int
        Number of LLA steps to take.

    stop_crit: str
        Which stopping criterion to use. Must be one of ['x_max', 'x_L2', 'loss'].

        If stop_crit='x_max' then we use ||x_new - x_prev||_max.

        If stop_crit='x_L2' then we use ||x_new - x_prev||_2.

        If stop_crit='loss' then we use loss(x_prev) - loss(x_new).

    tol: float, None
        Numerical value for stopping criterion. If None, then we will not use a stopping criterion.

    rel_crit: bool
        Should the tolerance be computed on a relative scale e.g. stop if ||x_new - x_prev||  <= tol * (||x_prev|| + epsilon).

    tracking_level: int
        How much optimization data to store at each step. Lower values means less informationed is stored.

    verbosity: int
        How much information to print out. Lower values means less print out.

    Output
    ------
    solution, solution_upv, sp_other_data, opt_info

    solution: array-like
        The solution of the penalized variable.

    solution_upv: None, array-like
        The solution of the unpenalized variable.

    sp_other_data:
        Other data output by the subproblem solver.

    opt_info: dict
        Data tracked during the optimization procedure e.g. the loss function.

    References
    ----------
    Fan, J., Xue, L. and Zou, H., 2014. Strong oracle optimality of folded concave penalized estimation. Annals of statistics, 42(3), p.819.
    """

    ######################
    # format initializer #
    ######################

    current = deepcopy(init)
    current_upv = deepcopy(init_upv)
    T = transform(current)

    # check stopping criteria
    if tol is None:
        stop_crit = None
    assert stop_crit is None or stop_crit in ['x_max', 'x_L2', 'loss']

    if stop_crit is not None:
        if current_upv is not None:
            prev = deepcopy(safe_concat(current, current_upv))
        else:
            prev = deepcopy(current)

    ##############################
    # optimization data tracking #
    ##############################
    history = {}

    # if we are using the loss tracking criteria then track the loss
    if stop_crit == 'loss' and tracking_level == 0:
        tracking_level = 1

    if tracking_level >= 1:
        if objective is None:
            raise ValueError("The objective function must be provided")

        history['objective'] = [objective(value=current, upv=current_upv)]

    if tracking_level >= 2:
        if stop_crit in ['x_max', 'x_L2']:
            history['x_diff'] = []

    #################
    # Run algorithm #
    #################

    start_time = time()

    step = 0  # in case max_steps = 0
    stop = False
    for step in range(int(max_steps)):

        if verbosity >= 1:
            print("Step {}, {:1.2f} seconds after start".
                  format(step + 1, time() - start_time))

        ###############
        # Make update #
        ###############

        # setup initialization for subproblem solver
        if sp_init is None:
            sp_init = current

        if sp_upv_init is None:
            sp_upv_init = current_upv

        # majorize concave penalty
        # T has already been computed as T = transform(current)
        weights = penalty_func.grad(T)

        # solve weighted Lasso problem
        current, current_upv, sp_other_data = \
            sub_prob.solve(weights=weights,
                           sp_init=sp_init,
                           sp_upv_init=sp_upv_init,
                           sp_other_data=sp_other_data)

        ############################################
        # check stopping conditions and track data #
        ############################################

        T = None  # tells us to compute T below

        # track objective function data
        if tracking_level >= 1:
            history['objective'].\
                append(objective(value=current, upv=current_upv))

        if stop_crit in ['x_max', 'x_L2']:

            if current_upv is not None:
                _current = safe_concat(current, current_upv)
            else:
                _current = current

            # check x difference stopping criterion
            stop, diff_norm = check_no_change(current=_current, prev=prev,
                                              tol=tol, rel_crit=rel_crit,
                                              norm=stop_crit[-3:]  # max or L2
                                              )

            if tracking_level >= 2:
                history['x_diff'].append(diff_norm)

        elif stop_crit == 'loss':
            current = history['objective'][-1]
            prev = history['objective'][-2]

            # check loss change criterion
            stop = check_decreasing_loss(current=current, prev=prev,
                                         tol=tol, rel_crit=rel_crit,
                                         on_increase='ignore')

        # # x change criteria
        # if xtol is not None:
        #     if current_upv is not None:
        #         _current = safe_concat(current, current_upv)
        #     else:
        #         _current = current

        #     x_stop, diff_norm = check_no_change(current=_current,
        #                                         prev=prev,
        #                                         tol=xtol,
        #                                         norm='max')

        #     if tracking_level >= 2:
        #         opt_info['diff_norm'].append(diff_norm)

        # if tracking_level >= 1:
        #     # objective function change criterion
        #     # if the objective has stopped decreasing we are done!
        #     obj_stop = check_decreasing_loss(current=opt_info['obj'][-1],
        #                                      prev_=opt_info['obj'][-2],
        #                                      tol=atol, rel_tol=rtol,
        #                                      on_increase='ignore')

        # maybe stop the algorithm
        if stop:
            break
        else:

            # set prev for next interation
            if stop_crit in ['x_max', 'x_L2']:
                prev = deepcopy(_current)

            # compute transform for next iteration if it has not already
            # been computed
            if T is None:
                T = transform(current)

    opt_info = {'runtime': time() - start_time,
                'history': history,
                'stop_crit': stop_crit,
                'stop': stop,
                'step': step}

    return current, current_upv, sp_other_data, opt_info


class WeightedProblemSolver(object):

    def solve(self, weights, sp_init=None,
              sp_upv_init=None, other_data=None):
        """
        Solves the weighted subproblem.

        Parameters
        ----------
        weights: array-like
            Weights for the weighted sup-problem.

        sp_init: None, array-like
            (Optional) Subproblem initialization for the penalized variable.

        sp_upv_init: None, array-like
            (Optional) Subproblem initialization for the unpenalized variable.

        other_data
            (Optional) Subproblem initialization for other data e.g. dual variables.

        Output
        ------
        solution, upv_solution, other_data
        """
        raise NotImplementedError
