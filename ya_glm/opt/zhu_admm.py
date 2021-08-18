from copy import deepcopy
from scipy.sparse import diags
import numpy as np
from time import time

from ya_glm.linalg_utils import leading_sval
from ya_glm.opt.utils import euclid_norm

# TODO: handle matrix shaped parameters
def solve(g1, g2, A1, A2,
          primal_init=None, dual_init=None,
          D_mat='prop_id',
          rho=1,
          rho_update=True,
          atol=1e-4,
          rtol=1e-4,
          eta=2,
          mu=10,
          max_iter=1000,
          tracking_level=0
          ):
    """
    Uses ADMM algorithm described in Section 2.4 of (Zhu, 2017) to solve a problem of the form

    min_theta g1(A_1 theta) + g2(A_2 theta)

    We only need assume g1 and g2 have easy to evaluate proximal operators.

    Parameters
    ----------
    g1, g2: ya_glm.opt.base.Func
        The two functions that make up the obejctive. Both must implement conj_prox which evaluates the proxial operator of the conjugate funtion, which is easily obtained from the proximal operator of the original function via Moreau's identity.

    A1, A2: array-like
        The two matrices in the objective function. Both matrices must have same number of columns, d.

    primal_init: None, array-like shape (d, )
        Optinal initialization for primal variable.

    dual_init: None, list list of of array-like.
        Optional initialization for the dual variables.
        The first list is the dual variables; the second list is the dual_bar variables.
        The first dual variable has shape (n_row(A_2), ) and the second
        has shape (n_row(A_2), ).

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

    Output
    ------
    solution, opt_data, admm_data

    solution: array-like
        The solution.

    opt_data: dict
        Opimization tracking data e.g. the dual/primal residual history.

    admm_data: dict
        Data related to ADMM e.g. the dual variables.

    References
    ----------
    Zhu, Y., 2017. An augmented ADMM algorithm with application to the generalized lasso problem. Journal of Computational and Graphical Statistics, 26(1), pp.195-204.
    """

    start_time = time()

    # shape data
    n_row_1 = A1.shape[0]
    n_row_2 = A2.shape[0]
    d = A1.shape[1]
    assert A2.shape[1] == d

    ########################
    # initialize variables #
    ########################
    # primal = np.zeros(d)
    # dual_1 = np.zeros(n_row_1)
    # dual_1_bar = np.zeros(n_row_1)
    # dual_2 = np.zeros(n_row_2)
    # dual_2_bar = np.zeros(n_row_2)

    if primal_init is None:
        primal = np.zeros(d)
    else:
        primal = deepcopy(primal_init)

    # dual variables
    if dual_init is not None:
        dual_1, dual_2 = dual_init[0]
        dual_1_bar, dual_2_bar = dual_init[1]

    else:
        # technically this initializes from 0 and takes one ADMM step
        dual_1 = g1.prox(rho * A1 @ primal, step=rho)
        dual_2 = g2.prox(rho * A2 @ primal, step=rho)

        dual_1_bar = 2 * dual_1
        dual_2_bar = 2 * dual_2

    # make sure we have correct shapes
    assert dual_1.shape[0] == n_row_1
    assert dual_2.shape[0] == n_row_2

    dual_cat = np.concatenate([dual_1, dual_2])
    dual_cat_prev = deepcopy(dual_cat)

    #################################
    # setup D matrix and A matrices #
    #################################

    if D_mat == 'prop_id':
        D_mat = DMatrixPropId()
    elif D_mat == 'diag':
        D_mat = DMatrixDiag()

    D_mat.setup(A1=A1, A2=A2)

    # Other setup
    # TODO: represent this lazily to avoid copying
    # A = np.vstack([A1, A2])
    A_mat = AMat(A1=A1, A2=A2)

    ##########################
    # setup history tracking #
    ##########################
    history = {}
    if tracking_level >= 1:
        history['primal_resid'] = []
        history['dual_resid'] = []
        history['rho'] = [rho]

        history['primal_tol'] = []
        history['dual_tol'] = []
    if tracking_level >= 2:
        history['primal'] = [primal]

    for it in range(int(max_iter)):

        # primal update
        primal_new = primal - \
            (1 / rho) * D_mat.inv_prod(A1.T @ dual_1_bar + A2.T @ dual_2_bar)

        # update dual variables
        dual_1_new = g1.conj_prox(rho * (A1 @ primal_new) + dual_1, step=rho)
        dual_2_new = g2.conj_prox(rho * (A2 @ primal_new) + dual_2, step=rho)

        dual_1_bar_new = 2 * dual_1_new - dual_1
        dual_2_bar_new = 2 * dual_2_new - dual_2

        # check stopping
        dual_cat_new = np.concatenate([dual_1_new, dual_2_new])
        primal_resid_norm = euclid_norm(dual_cat_new - dual_cat) / rho

        dual_resid = rho * A_mat.AtA_prod(primal_new - primal) + \
            A_mat.At_prod(2 * dual_cat - dual_cat_prev - dual_cat_new)

        dual_resid_norm = euclid_norm(dual_resid)

        # check stopping criteria
        # TODO: the relative part is not quite right, but I can't quite tell what it should be from the paper. Probably need to stare at it longer
        primal_tol = np.sqrt(A_mat.shape[0]) * atol + \
            rtol * euclid_norm(A_mat.A_prod(primal))

        dual_tol = np.sqrt(A_mat.shape[1]) * atol + \
            rtol * euclid_norm(A_mat.At_prod(dual_cat))

        # possibly track history
        if tracking_level >= 1:
            history['primal_resid'].append(primal_resid_norm)
            history['dual_resid'].append(dual_resid_norm)

            history['rho'].append(rho)

            # TODO-DEBUG: remove these
            history['primal_tol'].append(primal_tol)
            history['dual_tol'].append(dual_tol)

        if tracking_level >= 2:
            history['primal'].append(primal_new)

        if primal_resid_norm <= primal_tol and dual_resid_norm <= dual_tol:
            break

        # update variables if not stopping
        primal = primal_new
        dual_1 = dual_1_new
        dual_2 = dual_2_new
        dual_cat_prev = deepcopy(dual_cat)
        dual_cat = dual_cat_new
        dual_1_bar = dual_1_bar_new
        dual_2_bar = dual_2_bar_new

        # update rho
        # TODO: dont do every iteration
        if rho_update:
            primal_ratio = (primal_resid_norm / primal_tol)
            dual_ratio = (dual_resid_norm / dual_tol)
            if primal_ratio >= mu * dual_ratio:
                rho *= eta

            elif dual_ratio >= mu * primal_ratio:
                rho /= eta

    #################
    # Format output #
    #################
    # optimizaton data
    opt_data = {'iter': it,
                'runtime': time() - start_time,
                'history': history
                }

    # other data
    admm_data = {'dual_vars': [[dual_1_new, dual_2_new],
                               [dual_1_bar_new, dual_2_bar_new]],
                 'rho': rho,
                 'D_mat': D_mat
                 }

    return primal_new, opt_data, admm_data


def solve_path(prob_data_iter, **kws):
    """
    An iterator that computes the solution path over a sequence of problems using warm starts.

    Parameters
    ----------
    prob_data_iter: iterator
        Iterator yielding the sequence of problems to solve.
        Each element should be the tuple (g1, g2, A1, A2).

    **kws:
        Keyword arguments to ya_glm.opt.zhu_admm.solve

    Output
    ------
    soln, opt_data, admm_data
    """

    for (g1, g2, A1, A2) in prob_data_iter:
        soln, opt_data, admm_data = solve(g1=g1, g2=g2, A1=A1, A2=A2,
                                          **kws)

        yield soln, opt_data, admm_data

        kws['dual_init'] = admm_data['dual_vars']
        kws['rho'] = admm_data['rho']
        kws['D_mat'] = admm_data['D_mat']

        kws['primal_init'] = soln


class DMatrix:
    def setup(self, A1, A2):
        """
        Sets up the D matrix from the A1, A2 matrices.

        Parameters
        ----------
        A1, A2: array-like
            The two matrices in the objective function.

        Output
        ------
        self
        """
        pass

    def inv_prod(self, v):
        """
        Computes the product D_mat^{-1} @ v

        Parameters
        ----------
        v: array-like, shape (d, )
            The vector to mulitply by.
        Output
        ------
        D^{-1} v
        """
        pass


class DMatrixPropId(DMatrix):
    """
    Represents the matrix ||A||_op^2 * I_d
    """
    def setup(self, A1, A2):
        """
        Sets up the D matrix from the A1, A2 matrices.

        Parameters
        ----------
        A1, A2: array-like
            The two matrices in the objective function.

        Output
        ------
        self
        """
        AtA = A1.T @ A1 + A2.T @ A2
        self.sval_sq = leading_sval(AtA)
        # self.val = leading_sval(A1) + leading_sval(A2)

    def inv_prod(self, v):
        """
        Computes the product D_mat^{-1} @ v

        Parameters
        ----------
        v: array-like, shape (d, )
            The vector to mulitply by.
        Output
        ------
        D^{-1} v
        """
        return v / self.sval_sq


class DMatrixDiag(DMatrix):
    """
    Represents the diagonal matrix whose diagonal elements are given by sum_{j=1}^d |A^TA|_{ij}.
    """
    def setup(self, A1, A2):
        """
        Sets up the D matrix from the A1, A2 matrices.

        Parameters
        ----------
        A1, A2: array-like
            The two matrices in the objective function.

        Output
        ------
        self
        """
        AtA = A1.T @ A1 + A2.T @ A2
        row_sums = abs(AtA).sum(axis=1)

        self.diag_mat_inv = diags(1 / row_sums)

    def inv_prod(self, v):
        """
        Computes the product D_mat^{-1} @ v

        Parameters
        ----------
        v: array-like, shape (d, )
            The vector to mulitply by.
        Output
        ------
        D^{-1} v
        """
        return self.diag_mat_inv @ v


class DMatrixAtA(DMatrix):
    """
    D = A.T @ A
    """
    def setup(self, A1, A2):
        """
        Sets up the D matrix from the A1, A2 matrices.

        Parameters
        ----------
        A1, A2: array-like
            The two matrices in the objective function.

        Output
        ------
        self
        """
        A = np.vstack([A1, A2])
        self.AtA_inv = np.linalg.pinv(A.T @ A)

    def inv_prod(self, v):
        """
        Computes the product D_mat^{-1} @ v

        Parameters
        ----------
        v: array-like, shape (d, )
            The vector to mulitply by.
        Output
        ------
        D^{-1} v
        """
        return self.AtA_inv @ v


class AMat:
    """
    Represents the vertical concatenation of two matrices.

    Parameters
    ----------
    A1: array-like, (n_rows_1, n_cols)
        The first matrix.

    A2: array-like, (n_rows_2, n_cols)
        The second matrix.

    explicit_AtA: str, bool
        Whether or not to explicitly compute A.T @ A, which may save computational time but will take more memory. If default, will explicitly compute this if the number of columns significantly exceedds the number of rows.
    """

    def __init__(self, A1, A2, explicit_AtA='default'):
        self.A1 = A1
        self.A2 = A2

        self.n_rows_1 = A1.shape[0]
        self.n_rows_2 = A2.shape[0]
        self.shape = (self.n_rows_1 + self.n_rows_2, A1.shape[1])

        if explicit_AtA == 'default':

            if self.shape[0] < 10 * self.shape[1]:
                # lets only use more memory if it really really makes sense to
                self.explicit_AtA = False
            else:
                self.explicit_AtA = True

        assert type(self.explicit_AtA) == bool

        if explicit_AtA:
            self.AtA = np.array(A1.T @ A1 + A2.T @ A2)

    def AtA_prod(self, v):
        """
        Represents  A.T @ A @ v
        Parameteres
        -----------
        v: array-like, (n_cols, )
            The input vector.

        Output
        ------
        out: array-like, (n_cols, )
            out = A.T @ A @ v
        """
        if self.explicit_AtA:

            # n_cols ** 2
            return self.AtA @ v

        else:
            # n_rows x n_cols
            return self.A1.T @ (self.A1 @ v) + self.A2.T @ (self.A2 @ v)

    def At_prod(self, v):
        """
        Represents  A.T v

        Parameteres
        -----------
        v: array-like, (n_rols, )
            The input vector.

        Output
        ------
        out: array-like, (n_cols, )
            out = A.T @ v
        """
        p1 = self.A1.T @ v[:self.n_rows_1]
        p2 = self.A2.T @ v[self.n_rows_1:]

        return p1 + p2

    def A_prod(self, v):
        """
        Represents  A v

        Parameteres
        -----------
        v: array-like, (n_cols, )
            The input vector.

        Output
        ------
        out: array-like, (n_rows, )
            out = A @ v
        """
        p1 = self.A1 @ v
        p2 = self.A2 @ v
        return np.hstack([p1, p2])
