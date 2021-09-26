import numpy as np
from scipy.sparse import csr_matrix

from ya_glm.solver.base import GlmSolverWithPath
from ya_glm.autoassign import autoassign

from ya_glm.config.loss import get_loss_config
from ya_glm.config.constraint import get_constraint_config
from ya_glm.config.penalty import get_penalty_config
from ya_glm.config.base_params import get_base_config

from ya_glm.opt.zhu_admm import solve
from ya_glm.opt.from_config.input_loss import get_glm_input_loss
from ya_glm.opt.from_config.mat_and_func import get_mat_and_func
from ya_glm.opt.from_config.penalty import get_penalty_func

from ya_glm.config.penalty import NoPenalty
from ya_glm.opt.utils import decat_coef_inter_vec, decat_coef_inter_mat, \
    process_zero_init
from ya_glm.utils import is_multi_response, get_shapes_from
from ya_glm.sparse_utils import safe_hstack


class ZhuADMM(GlmSolverWithPath):
    """
    Solves a penalized GLM problem using the augmented ADMM algorithm of (Zhu, 2017).

    Parameters
    ----------
    D_mat: str, ya_glm.addm.addm.DMatrix
        The D matrix. If str, must be one of ['prop_id', 'diag'].
        If 'prop_id' then D will be ||A||_op * I_d.
        If 'diag', then D will be the diagonal matrix whose ith element is given by sum_{j=1}^d |A^TA|_{ij}.

    rho: float
        The ADMM penalty parameter.

    rho_update: bool
        Whether or not to adpatively update the rho parameter.

    atol, rtol: float
        The absolute and relative stopping criteria.

    eta: float
        Amount to increase/decrease rho by.

    mu: float
        Parameter for deciding whether or not to increase rho. See (15) from (Zhu, 2017).

    max_iter: int
        Maximum number of iterations.

    tracking_level: int
        How much data to track.

    References
    ----------
    Zhu, Y., 2017. An augmented ADMM algorithm with application to the generalized lasso problem. Journal of Computational and Graphical Statistics, 26(1), pp.195-204.
    """

    @autoassign
    def __init__(self,
                 D_mat='diag',
                 rho=1,
                 rho_update=True,
                 atol=1e-4,
                 rtol=1e-4,
                 eta=2,
                 mu=10,
                 max_iter=1000,
                 tracking_level=0): pass

    @classmethod
    def is_applicable(self, loss, penalty=None, constraint=None):
        """
        Determines whether or not this problem can be solved the ADMM algorithm i.e. if it is in the form of

        min L(X @ coef) + p(mat @ coef)

        where both L and p are proximable.

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

        if constraint is not None:
            raise NotImplementedError("TODO add this!")

        # pull out base configs
        loss = get_base_config(get_loss_config(loss))
        penalty = get_base_config(get_penalty_config(penalty))
        if constraint is not None:
            constraint = get_base_config(get_constraint_config(constraint))

        # make fake data just for getting functions
        # X = np.zeros((3, 2))
        y = np.zeros(3)
        n_features = 2

        # get g1 and g2 functions
        g1 = get_glm_input_loss(config=loss, y=y)
        g2_config = get_mat_and_func(config=penalty, n_features=n_features)[1]
        g2 = get_penalty_func(g2_config, n_features=n_features)

        if g1.is_proxiable and g2.is_proximable:
            return True
        else:
            return False

    def setup(self, X, y, loss, penalty, constraint=None,
              fit_intercept=True, sample_weight=None):
        """
        Sets up anything the solver needs.
        """
        # make sure FISTA is applicable
        if not self.is_applicable(loss, penalty, constraint):
            raise ValueError("ADMM is not applicable to "
                             "loss={}, penalty={}, constrain={}".
                             format(loss, penalty, constraint))

        # store some data we need in solve()
        self.is_mr_ = is_multi_response(y)
        self.fit_intercept_ = fit_intercept
        # self.penalty_config_ = penalty
        self.coef_shape_, self.intercept_shape_ = get_shapes_from(X=X, y=y)
        self.n_features_ = X.shape[1]

        # and identity respectively
        # get the linear transform matrix and the transformed penalty
        if penalty is None:
            penalty = NoPenalty()
        self.A2_, self.g2_config_ = get_mat_and_func(config=penalty,
                                                     n_features=X.shape[1])
        self.g2_ = get_penalty_func(config=self.g2_config_,
                                    n_features=self.n_features_)
        # TODO: let g2/A2 be None for the zero function

        # set the X transformation matrix
        if fit_intercept:
            ones_col = np.ones(X.shape[0]).reshape(-1, 1)
            # TODO: make ones a linear operator if X is sparse
            self.A1_ = safe_hstack([ones_col, X])  # safely addresses sparse X
        else:
            self.A1_ = X

        # set the loss function
        self.g1_ = get_glm_input_loss(config=loss,
                                      y=y,
                                      sample_weight=sample_weight)

    def update_penalty(self, **params):
        """
        Updates the penalty parameters.
        """
        # TODO: this only updates the penalty that is applied to the
        # linear transformed coefficient. This is a bit misleading.
        self.g2_config_.set_params(**params)
        self.g2_ = get_penalty_func(config=self.g2_config_,
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

        other_init: None, dicts
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

        ########################
        # process initializers #
        ########################
        coef_init, intercept_init = \
            process_zero_init(coef_shape=self.coef_shape_,
                              intercept_shape=self.intercept_shape_,
                              coef_init=coef_init,
                              intercept_init=intercept_init,
                              fit_intercept=self.fit_intercept_)

        if self.fit_intercept_:
            primal_init = np.concatenate([[intercept_init], coef_init])
        else:
            primal_init = coef_init

        # other init is a dict here
        if other_init is None:
            other_init = {}

        ###########################################
        # Possibly add intercept to penalty terms #
        ###########################################

        if self.fit_intercept_:
            # add a zero column to A2
            zero_col = csr_matrix((self.A2_.shape[0], 1))
            A2 = safe_hstack([zero_col, self.A2_])

        else:
            A2 = self.A2_

        ###########################
        # solve problem with ADMM #
        ###########################

        # merge other_init with solver keyword arguments
        kws = {**other_init}
        for k, v in self.get_solve_kws().items():
            # only add this kws if it is not given by other_init
            if k not in kws:
                kws[k] = v

        # rename dual_vars to dual_init
        for k in list(kws.keys()):
            if k == 'dual_vars':
                kws['dual_init'] = kws.pop('dual_vars')

        soln, admm_data, opt_info = solve(g1=self.g1_,
                                          g2=self.g2_,
                                          A1=self.A1_,
                                          A2=A2,
                                          primal_init=primal_init,
                                          **kws)

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

        return soln, admm_data, opt_info
