from copy import deepcopy

from ya_glm.GlmSolver import GlmSolver
from ya_glm.autoassign import autoassign
from ya_glm.opt.glm_loss.get import get_glm_input_loss


class ADMMSolver(GlmSolver):
    """
    Solves a penalized GLM problem using the augmented ADMM algorithm from (Zhu, 2017)

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
                 D_mat='prop_id',
                 rho=1,
                 rho_update=True,
                 atol=1e-4,
                 rtol=1e-4,
                 eta=2,
                 mu=10,
                 max_iter=1000,
                 tracking_level=0): pass

    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def _get_avail_losses(self):
        return ['lin_reg', 'huber',
                'log_reg', 'multinomial',
                'poisson', 'quantile']

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


        # initialize primal
        if fit_intercept:
            if coef_init is not None and intercept_init is not None:
                primal_init = CONCAT(coef_init, intercept_init)

            else:
                primal_init = coef_init

        g1 = get_glm_input_loss(y=y, loss=loss,
                                sample_weight=sample_weight)

        g2 =

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




    def has_path_algo(self, loss, penalty):
        """
        Yes this solver has an available path algorithm!
        """
        return True
