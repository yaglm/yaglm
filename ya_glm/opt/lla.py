from copy import deepcopy
from time import time

from ya_glm.opt.stopping import check_decreasing_loss, check_no_change
from ya_glm.opt.utils import safe_concat


def solve_lla(sub_prob, penalty_func,
              init, init_upv=None,
              sp_init=None, sp_upv_init=None, sp_other_data=None,
              transform=abs, objective=None,
              n_steps=1, xtol=1e-4, atol=None, rtol=None,
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

    if xtol is not None:
        if current_upv is not None:
            prev = deepcopy(safe_concat(current, current_upv))
        else:
            prev = deepcopy(current)

    ##############################
    # optimization data tracking #
    ##############################
    opt_info = {}

    if (atol is not None or rtol is not None) and tracking_level == 0:
        tracking_level = 1

    if tracking_level >= 1:
        if objective is None:
            raise ValueError("The objective function must be provided")

        opt_info['obj'] = [objective(value=current, upv=current_upv)]

    if tracking_level >= 2:
        if xtol is not None:
            opt_info['diff_norm'] = []  # difference between successive iterates

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
            opt_info['obj'].append(objective(value=current, upv=current_upv))

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
                opt_info['diff_norm'].append(diff_norm)

        if tracking_level >= 1:
            # objective function change criterion
            # if the objective has stopped decreasing we are done!
            obj_stop = check_decreasing_loss(current_loss=opt_info['obj'][-1],
                                             prev_loss=opt_info['obj'][-2],
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

    opt_info['runtime'] = time() - start_time
    opt_info['final_step'] = step
    opt_info['obj_stop'] = obj_stop
    opt_info['x_stop'] = x_stop

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
