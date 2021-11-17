import numpy as np

from yaglm.solver.base import GlmSolverWithPath
from yaglm.autoassign import autoassign
from yaglm.opt.algo.fista import solve_fista


from yaglm.opt.from_config.loss import get_glm_loss_func
from yaglm.opt.from_config.penalty import get_penalty_func, wrap_intercept
from yaglm.opt.split_smooth_and_non_smooth import split_smooth_and_non_smooth
from yaglm.opt.from_config.constraint import get_constraint_func

from yaglm.opt.base import Sum
from yaglm.opt.utils import decat_coef_inter_vec, decat_coef_inter_mat
from yaglm.utils import is_multi_response


class FISTA(GlmSolverWithPath):
    """
    Solves a penalized GLM problem using the FISTA algorithm.

    Parameters
    ----------
    max_iter: int
        Maximum number of iterations.

    stop_crit: str
        Which stopping criterion to use. Must be one of ['x_max', 'x_L2', 'loss'].

        If stop_crit='x_max' then we use ||x_new - x_prev||_max.

        If stop_crit='x_L2' then we use ||x_new - x_prev||_2.

        If stop_crit='loss' then we use loss(x_prev) - loss(x_new).

    tol: float, None
        Numerical value for stopping criterion. If None, then we will not use a stopping criterion.

    rel_crit: bool
        Should the tolerance be computed on a relative scale e.g. stop if ||x_new - x_prev||  <= tol * (||x_prev|| + epsilon).

    bt_max_steps: int
        Maximum number of backtracking steps to take.

    bt_shrink: float
        How much to shrink the step size in each backtracking step. Should lie strictly in the unit interval.

    bt_grow: float, None
        (Optional) How much to grow the step size each iteraction when using backgracking.

    accel: bool
        Whether or not to use FISTA acceleration.

    restart: bool
        Whether or not to restart the acceleration scheme. See (13) from https://bodono.github.io/publications/adap_restart.pdf
        for the strategy we employ.

    tracking_level: int
        How much data to track.

    References
    ----------
    Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1), pp.183-202.
    """

    @autoassign
    def __init__(self,
                 max_iter=1000,
                 tol=1e-5, rel_crit=False, stop_crit='x_max',
                 bt_max_steps=20,
                 bt_shrink=0.5,
                 bt_grow=1.58,  # 10**.2
                 accel=True,
                 restart=True,
                 tracking_level=0): pass

    @classmethod
    def _is_applicable(self, loss, penalty=None, constraint=None):
        """
        Determines whether or not this problem can be solved by FISTA i.e. if it is in the form

        min L(coef) + p(coef)

        where L is smooth and p is proximable.

        Parameters
        ----------
        loss: LossConfig
            The loss.

        penalty: None, PenaltyConfig
            The penalty.

        constraint: None, ConstraintConfig

        Output
        ------
        is_applicable: bool
            Wheter or not this solver can be used.
        """

        # make fake data just for getting functions
        X = np.zeros((3, 2))
        y = np.zeros(3)
        n_features = 2

        # get functions
        loss_func = get_glm_loss_func(config=loss, X=X, y=y)
        penalty_func = get_penalty_func(config=penalty, n_features=n_features)

        # split penalty into smooth and non-smooth
        smooth_pen, non_smooth_pen = split_smooth_and_non_smooth(penalty_func)

        if constraint is not None:
            constraint_func = get_constraint_func(constraint)

        # don't currently support penalty with constraint
        if penalty is not None and constraint is not None:
            # TODO: there are some special cases where this will work
            return False

        # don't support non-smooth loss functions
        elif not loss_func.is_smooth:
            # TODO: perhaps modify to work with smooth penalty and
            # non-smooth loss e.g. quantile with ridge
            return False

        # don't support non-smoooth, non-proximable penalties
        elif non_smooth_pen is not None and not non_smooth_pen.is_proximable:
            return False

        # don't support non-proximable constraints
        elif constraint is not None and not constraint_func.is_proximable:
            return False

        else:
            return True

    def setup(self, X, y, loss, penalty, constraint=None,
              fit_intercept=True, sample_weight=None):
        """
        Sets up anything the solver needs.
        """
        # make sure FISTA is applicable
        if not self.is_applicable(loss, penalty, constraint):
            raise ValueError("FISTA is not applicable to "
                             "loss={}, penalty={}, constrain={}".
                             format(loss, penalty, constraint))

        self.is_mr_ = is_multi_response(y)
        self.fit_intercept_ = fit_intercept
        self.penalty_config_ = penalty
        self.n_features_ = X.shape[1]

        #################
        # Loss function #
        #################

        # get the loss function
        self.loss_func_ = get_glm_loss_func(config=loss, X=X, y=y,
                                            fit_intercept=fit_intercept,
                                            sample_weight=sample_weight)

        ##########################
        # Penalty and constraint #
        ##########################

        self.penalty_func_ = None
        self.constraint_func_ = None

        if penalty is not None:
            self.penalty_func_ = get_penalty_func(config=self.penalty_config_,
                                                  n_features=self.n_features_)
        if constraint is not None:
            self.constraint_func_ = get_constraint_func(config=constraint)

    def update_penalty(self, **params):
        """
        Updates the penalty parameters.
        """

        self.penalty_config_.set_params(**params)
        self.penalty_func_ = get_penalty_func(config=self.penalty_config_,
                                              n_features=self.n_features_)

    def solve(self, coef_init=None, intercept_init=None, other_init=None):
        """
        Solves the optimization problem.

        Parameters
        ----------
        coef_init: None, array-like
            (Optional) Initialization for the coefficient.

        intercept_init: None, array-like
            (Optional) Initialization for the intercept.

        other_init: None, array-like
            (Optional) Initialization for other optimization data e.g. dual variables.

        Output
        ------
        soln, other_data, opt_info

        soln: dict of array-like
            The coefficient/intercept solutions,

        other_data: dict
            Other optimzation output data e.g. dual variables.

        opt_info: dict
            Optimization information e.g. number of iterations, runtime, etc.
        """

        #########
        # Setup #
        #########

        # split penalty into smooth and non-smooth parts
        smooth_pen, non_smooth_pen = \
            split_smooth_and_non_smooth(self.penalty_func_)

        # maybe add an intercept to the penalty
        if smooth_pen is not None:
            smooth_pen = wrap_intercept(func=smooth_pen,
                                        fit_intercept=self.fit_intercept_,
                                        is_mr=self.is_mr_)

        if non_smooth_pen is not None:
            non_smooth_pen = wrap_intercept(func=non_smooth_pen,
                                            fit_intercept=self.fit_intercept_,
                                            is_mr=self.is_mr_)

        # set smooth/non-smooth functions
        # set smooth/non-smooth functions
        if smooth_pen is not None:
            smooth_func = Sum([self.loss_func_, smooth_pen])
        else:
            smooth_func = self.loss_func_

        if self.constraint_func_ is not None:
            assert non_smooth_pen is None
            non_smooth_func = self.constraint_func_
        else:
            non_smooth_func = non_smooth_pen

        # setup step size/backtracking
        if smooth_func.grad_lip is not None:
            # use Lipchtiz constant if it is available
            step = 'lip'
            backtracking = False
        else:
            step = 1  # TODO: perhaps smarter base step size?
            backtracking = True

        # setup initial value
        if coef_init is None or  \
                (self.fit_intercept_ and intercept_init is None):
            init_val = self.loss_func_.default_init()
        else:
            init_val = self.loss_func_.\
                cat_intercept_coef(intercept_init, coef_init)

        ############################
        # solve problem with FISTA #
        ############################
        soln, out = solve_fista(smooth_func=smooth_func,
                                init_val=init_val,
                                non_smooth_func=non_smooth_func,
                                step=step,
                                backtracking=backtracking,
                                **self.get_solve_kws())

        # format output
        if self.fit_intercept_:
            if self.is_mr_:
                coef, intercept = decat_coef_inter_mat(soln)
            else:
                coef, intercept = decat_coef_inter_vec(soln)
        else:
            coef = soln
            intercept = None

        soln = {'coef': coef, 'intercept': intercept}
        opt_data = None
        opt_info = out

        return soln, opt_data, opt_info
