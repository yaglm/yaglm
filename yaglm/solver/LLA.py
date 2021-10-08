from copy import deepcopy

from warnings import warn
from ya_glm.solver.base import GlmSolverWithPath
from ya_glm.opt.lla import solve_lla, WeightedProblemSolver
from ya_glm.opt.from_config.penalty import get_penalty_func, wrap_intercept
from ya_glm.opt.from_config.loss import get_glm_loss_func
from ya_glm.opt.utils import safe_concat
from ya_glm.config.penalty_utils import get_flavor_kind

from ya_glm.opt.from_config.lla import get_lla_nonconvex_func, \
    get_lla_subproblem_penalty, get_lla_transformer

from ya_glm.autoassign import autoassign
from ya_glm.utils import is_multi_response


class LLAFixedInit(GlmSolverWithPath):
    """
    Solves a foled concave penalized GLM problem using the LLA algorithm initialized from a specified starting point.

    Parameters
    ----------
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

    sp_solver: GlmSolver
        The solver to use for the penalized GLM subproblems.

    Attributes
    ----------
    sp_solver_: WeightedGlmProblemSolver
        The weighted subproblem solver.

    coef_init_lla_:
        The coefficient initializer for the LLA algorithm.

    intercept_init_lla_:
        The intercept initializer for the LLA algorithm.

    transform_:
        (Optional) The transformation applied to the coefficient.

    transf_penalty_func_:
        The non-convex function applied to the transformed coefficient.
    """
    @autoassign
    def __init__(self, n_steps=1, xtol=1e-4, atol=None, rtol=None,
                 tracking_level=0, verbosity=0): pass

    @classmethod
    def _is_applicable(self, loss, penalty=None, constraint=None):
        # perhaps make sure the flavor is non-convex?
        # or at least not adaptive
        return True

    def set_sp_solver(self, solver):
        self.sp_solver_ = WeightedGlmProblemSolver(solver=solver)

    def setup(self, X, y, loss, penalty, constraint=None,
              fit_intercept=True, sample_weight=None):

        kws = locals()
        kws.pop('self')

        if get_flavor_kind(penalty) not in ['non_convex', 'mixed']:
            warn("The LLA algorithm is being called on a "
                 "non-non-convex problem! Perhaps something silently failed.")

        self.penalty_config_ = deepcopy(penalty)

        # weighted subproblem solver
        self.sp_solver_.setup(**kws)

        # objective function evaulator
        kws.pop('constraint')
        self.objective_ = ObjectiveFunc(**kws)

        # set the coefficient transform
        self.transform_ = get_lla_transformer(self.penalty_config_)

        # the non-convex function applied to the transformed coefficient
        # this function is designed so we can update the LLA subprobelm's
        # penalty weights via self.solver_.update_penalty(**weights)
        self.transf_penalty_func_ = get_lla_nonconvex_func(self.penalty_config_)

    def update_penalty(self, **params):
        """
        Updates the penalty.
        """
        # TODO: i don't think we would ever update transform_

        # update the subproblem solver e.g. if we changed
        # a non-non-convex function's parameters
        self.sp_solver_.update_penalty(**params)

        # update the objective function
        self.objective_.update_penalty(**params)

        # update non-convex function applied to the transformed coefficient
        self.penalty_config_.set_params(**params)
        self.transf_penalty_func_ = get_lla_nonconvex_func(self.penalty_config_)

    @property
    def needs_fixed_init(self):
        return True

    def set_fixed_init(self, init_data):
        """
        Sets the fixed LLA initializer.

        Parameters
        ----------
        init_data: dict
            The initializer data; has keys ['coef', 'intercept'].
        """

        # where we initialize the coefficient and intercept
        self.coef_init_lla_ = init_data['coef']
        self.intercept_init_lla_ = init_data.get('intercept', None)

    def solve(self, coef_init=None, intercept_init=None, other_init=None):
        """
        Solves the LLA algorithm from a fixed starting point.

        Parameters
        ----------
        coef_init, intercept_init, other_init:
            (Optional) Initialization for the first LLA subproblem. Note this is not the initialization for entire the LLA algorithm, which is set TODO.

        Output
        ------
        soln, other_data, opt_info

        soln: dict of array-like
            The coefficient/intercept solutions.

        other_data:
            Other data output by the LLA algorithm e.g. dual variables.

        opt_info: dict
            Optimization information e.g. runtime, number of steps.
        """
        # TODO-THINK-THROUGH: the initialization is a bit misleading here. For all other solvers coef_init is where we initializer the entire optimization algorithm, but here it is only the initialization for the first LLA sub-problem. The advantage of this current version is that it allows us to do warm starts for the LLA algorithm.

        coef, intercept, sp_other_data, opt_info = \
            solve_lla(sub_prob=self.sp_solver_,
                      penalty_func=self.transf_penalty_func_,

                      # Actual LLA algorithm initialization
                      init=self.coef_init_lla_,
                      init_upv=self.intercept_init_lla_,

                      # warm start the subproblem solver
                      sp_init=coef_init,
                      sp_upv_init=intercept_init,
                      sp_other_data=other_init,

                      transform=self.transform_,
                      objective=self.objective_,

                      **self.get_solve_kws())

        return {'coef': coef, 'intercept': intercept}, sp_other_data, opt_info


class WeightedGlmProblemSolver(WeightedProblemSolver):
    """
    The solver for the weighted subproblems from the LLA algorithm. Also computes the objective function.

    Parameters
    ----------
    solver: GlmSolver
        A solver that solves the weighted GLM subproblems.

    Attributes
    ----------
    sp_solver_: Solver
        The weighted sub-problem solver.

    loss_func_: ya_glm.opt.base.Func
        The loss function -- used to compute the objective function.

    penalty_func_:ya_glm.opt.base.Func
        The non-convex penalty function -- used to compute the objective function.

    is_mr_: bool
        Whether or not the coefficient is a multi-response coefficient.

    fit_intercept_: bool
        Whether or not there is an intercept in the model.
    """
    def __init__(self, solver):
        self.solver = solver

    def setup(self, X, y, loss, penalty, constraint=None,
              fit_intercept=True, sample_weight=None):
        """
        Sets up anything the solver needs.
        """

        # setup solver with base convex penalty
        kws = locals()
        kws.pop('self')

        # penalty config for the overall problem
        self.penalty_config_ = penalty

        # set solver with convex penalty config
        sp_penalty = get_lla_subproblem_penalty(deepcopy(self.penalty_config_))

        kws['penalty'] = sp_penalty
        self.solver_ = deepcopy(self.solver)
        self.solver_.setup(**kws)
        # TODO: do we want a copy/clone here?

        self.fit_intercept_ = fit_intercept

    def update_penalty(self, **params):
        """
        updates the overall problem penalty
        """
        # TODO: document
        # this will only update any non non-convex penalty parameters that were changed
        self.penalty_config_.set_params(**params)
        sp_penalty = get_lla_subproblem_penalty(self.penalty_config_)
        self.solver_.update_penalty(**sp_penalty.get_params(deep=True))

    def solve(self, weights, sp_init=None,
              sp_upv_init=None, sp_other_data=None):
        """
        Solves the weighted subproblem.

        Parameters
        ----------
        weights: dict, array-like
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

        # update penalty weights
        self.solver_.update_penalty(**weights)

        soln, other_data, opt_info = \
            self.solver_.solve(coef_init=sp_init,
                               intercept_init=sp_upv_init,
                               other_init=sp_other_data)

        return soln['coef'], soln['intercept'], other_data


class ObjectiveFunc:

    def __init__(self, X, y, loss, penalty,
                 fit_intercept=True, sample_weight=None):

        # setup loss + penalty for computing loss function
        self.loss_func_ = get_glm_loss_func(X=X, y=y, config=loss,
                                            fit_intercept=fit_intercept,
                                            sample_weight=sample_weight)

        self.is_mr_ = is_multi_response(y)
        self.fit_intercept_ = fit_intercept
        self.n_features_ = X.shape[1]

        self.penalty_config_ = deepcopy(penalty)
        self.update_penalty()

    def update_penalty(self, **params):
        """
        Updates the penalty.
        """
        self.penalty_config_.set_params(**params)

        # the overall penalty function used for computing the objective function
        self.penalty_func_ = get_penalty_func(self.penalty_config_,
                                              n_features=self.n_features_)
        self.penalty_func_ = wrap_intercept(func=self.penalty_func_,
                                            fit_intercept=self.fit_intercept_,
                                            is_mr=self.is_mr_)

    def __call__(self, value, upv=None):
        """
        Evaluates the objective function.

        Parameters
        ----------
        value:
            The current value of the penalized variable.

        upv:
            (Optional) The current value of the unpenalized variable.

        Output
        ------
        obj: float
            The objective function value.
        """
        if self.fit_intercept_:
            # return self.glm_loss.eval(np.concatenate([[upv], value]))
            current = safe_concat(upv, value)
        else:
            current = value

        base_loss = self.loss_func_.eval(current)
        pen_loss = self.penalty_func_.eval(current)
        obj = base_loss + pen_loss

        return obj
