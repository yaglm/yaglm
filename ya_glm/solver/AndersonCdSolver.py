from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.sparse import issparse
from sklearn.utils import check_array

from andersoncd.penalties import L1, WeightedL1, L1_plus_L2
from andersoncd.solver import solver

from ya_glm.solver.utils import process_param_path
from ya_glm.solver.fake_intercept import center_Xy, \
    fake_intercept_via_centering

from ya_glm.sparse_utils import safe_norm
from ya_glm.GlmSolver import GlmSolver
from ya_glm.autoassign import autoassign


class AndersonCdSolver(GlmSolver):
    """
    Solves a penalized GLM problem using the andersoncd package (https://github.com/mathurinm/andersoncd).

    Parameters
    ----------
    fake_intercept: True
        Andersoncd does not allow an intercept, but we can fake it in the same way sklearn.linear_model.Lasso does via appropriate centering.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.
    """

    @autoassign
    def __init__(self, max_iter=20, max_epochs=50000,
                 p0=10, verbose=0, tol=1e-4, prune=0): pass

    def get_solve_kws(self):
        return deepcopy(self.__dict__)

    def _get_avail_losses(self):
        return ['lin_reg']

    # TODO:
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

        return solve_glm(X=X, y=y,
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
        Solves a sequence of penalized GLM problems. See docs for ya_glm.GlmSolver.
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


def solve_glm(X, y, loss,
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

              coef_init=None,
              intercept_init=None,

              fake_intercept=True,
              max_iter=20, max_epochs=50000,
              p0=10, verbose=0, tol=1e-4, prune=0):
    """
    Solves a penalized GLM using the andersoncd package (https://github.com/mathurinm/andersoncd).

    Parameters
    ----------
    fake_intercept: True
        Andersoncd does not allow an intercept, but we can fake it in the same way sklearn.linear_model.Lasso does via appropriate centering.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.
    """

    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False, accept_large_sparse=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    # if we want to fake the intercept
    if fit_intercept and fake_intercept:
        # center the data matrix and y data
        if max(X.mean(axis=0).max(), y.mean(axis=0).max()) < 1e-3:
            # hack to check if we have already performed centering
            # note this will not work for sample weights
            pre_pro_out = {'X_offset': np.zeros(X.shape[0]),
                           'y_offset': 0}
        else:
            X, y, pre_pro_out = center_Xy(X=X, y=y, copy=True)

    check_args(X=X, y=y, loss=loss,
               fit_intercept=fit_intercept,
               fake_intercept=fake_intercept,
               sample_weight=sample_weight,
               lasso_pen_val=lasso_pen_val,
               lasso_weights=lasso_weights,
               groups=groups,
               multi_task=multi_task,
               nuc=nuc,
               ridge_pen_val=ridge_pen_val,
               ridge_weights=ridge_weights,
               tikhonov=tikhonov)

    #######################################
    # setup initialization and other data #
    #######################################

    penalty = get_penalty(lasso_pen_val=lasso_pen_val,
                          lasso_weights=lasso_weights,
                          ridge_pen_val=ridge_pen_val)

    if coef_init is None:
        coef_init = np.zeros(X.shape[1], dtype=X.dtype)
        R = y.copy()
    else:
        p0 = max((coef_init != 0.).sum(), p0)
        R = y - X @ coef_init

    norms_X_col = safe_norm(X, axis=0)

    coef, obj_hist, kkt_max = solver(X=X, y=y,
                                     penalty=penalty,
                                     w=coef_init.copy(),
                                     R=R, norms_X_col=norms_X_col,
                                     max_iter=max_iter,
                                     max_epochs=max_epochs,
                                     p0=p0,
                                     tol=tol,
                                     verbose=verbose)

    opt_out = {'objective': obj_hist,
               'kkt_max': kkt_max}

    # fake intercept
    if fit_intercept and fake_intercept:
        intercept = fake_intercept_via_centering(coef, **pre_pro_out)
    else:
        intercept = None

    return coef, intercept, opt_out


def solve_glm_path(X, y, loss,
                   lasso_pen_seq=None, ridge_pen_seq=None,

                   check_decr=True, generator=True,

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

                   coef_init=None,
                   intercept_init=None,

                   fake_intercept=True,
                   max_iter=20, max_epochs=50000,
                   p0=10, verbose=0, tol=1e-4, prune=0):
    """
    Solves a sequence of penalized GLM using the andersoncd package (https://github.com/mathurinm/andersoncd).
    """

    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False, accept_large_sparse=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    # if we want to fake the intercept
    if fit_intercept and fake_intercept:
        # center the data matrix and y data
        if max(X.mean(axis=0).max(), y.mean(axis=0).max()) < 1e-3:
            # hack to check if we have already performed centering
            # note this will not work for sample weights
            pre_pro_out = {'X_offset': np.zeros(X.shape[0]),
                           'y_offset': 0}
        else:
            if issparse(X):
                raise NotImplementedError("safe_norm below will break"
                                          "Due to the way we center."
                                          "This should be fixable!")
            X, y, pre_pro_out = center_Xy(X=X, y=y, copy=True)

    param_path = process_param_path(lasso_pen_seq=lasso_pen_seq,
                                    ridge_pen_seq=ridge_pen_seq,
                                    check_decr=check_decr)

    if 'lasso_pen_val' in param_path:
        temp_lasso_pen_val = param_path['lasso_pen_val'][0]
    else:
        temp_lasso_pen_val = lasso_pen_val

    if 'ridge_pen_val' in param_path:
        temp_ridge_pen_val = param_path['ridge_pen_val'][0]
    else:
        temp_ridge_pen_val = ridge_pen_val

    check_args(X=X, y=y, loss=loss,
               fit_intercept=fit_intercept,
               fake_intercept=fake_intercept,
               sample_weight=sample_weight,
               lasso_pen_val=temp_lasso_pen_val,
               lasso_weights=lasso_weights,
               groups=groups,
               multi_task=multi_task,
               nuc=nuc,
               ridge_pen_val=temp_ridge_pen_val,
               ridge_weights=ridge_weights,
               tikhonov=tikhonov)

    #######################################
    # setup initialization and other data #
    #######################################

    if coef_init is None:
        coef = np.zeros(X.shape[1], dtype=X.dtype)
        R = y.copy()
    else:
        coef = coef_init.copy()
        R = y - X @ coef_init

    norms_X_col = safe_norm(X, axis=0)

    solve_verbose = verbose >= 2
    n_params = len(param_path)
    for i, params in enumerate(param_path):
        if verbose >= 1:
            print('Solving path parameter {}/{}'.format(i, n_params))

        # set penalty for this path
        penalty = get_penalty(lasso_weights=lasso_weights,
                              **params)

        # solve!
        p0 = max((coef != 0.).sum(), p0)

        coef, obj_hist, kkt_max = solver(X=X, y=y,
                                         penalty=penalty,
                                         w=coef,
                                         R=R,
                                         norms_X_col=norms_X_col,
                                         max_iter=max_iter,
                                         max_epochs=max_epochs,
                                         p0=p0,
                                         tol=tol,
                                         verbose=solve_verbose)

        # fake intercept
        if fit_intercept and fake_intercept:
            intercept = fake_intercept_via_centering(coef, **pre_pro_out)
        else:
            intercept = None

        # format output
        fit_out = {'coef': coef,
                   'intercept': intercept,
                   'opt_data':  {'objective': obj_hist,
                                 'kkt_max': kkt_max}}

        yield fit_out, params


def check_args(X, y,
               loss,
               fit_intercept,
               fake_intercept,
               sample_weight,

               lasso_pen_val,
               lasso_weights,
               groups,
               multi_task,
               nuc,
               ridge_pen_val,
               ridge_weights,
               tikhonov):

    #############################
    # check what is implemented #
    #############################

    if loss.name != 'lin_reg':
        raise NotImplementedError("{} is not supported".format(loss.name))

    if fit_intercept and not fake_intercept:
        raise NotImplementedError("exact intercept not supported")

    if sample_weight is not None:
        raise NotImplementedError("sample_weight not supported")

    if groups is not None:
        raise NotImplementedError("groups is not supported")

    if multi_task:
        raise NotImplementedError("multi_task is not yet supported")

    if nuc:
        raise NotImplementedError("nuc is not yet supported")

    if tikhonov is not None:
        raise NotImplementedError("tikhonov is not yet supported")

    if ridge_weights is not None:
        raise NotImplementedError("ridge_weights is not yet supported")

    if ridge_pen_val is not None and lasso_weights is not None:
        raise NotImplementedError("lasso_weights and ridge penalty not yet supported")


def get_penalty(lasso_pen_val=None, lasso_weights=None, ridge_pen_val=None):
    #############################
    # pre process penalty input #
    #############################

    if lasso_pen_val is None and lasso_weights is not None:
        lasso_pen_val = 1
    elif lasso_pen_val is None:
        lasso_pen_val = 0

    #################
    # Setup penalty #
    #################

    if lasso_weights is None:
        if ridge_pen_val is None:
            penalty = L1(alpha=lasso_pen_val)
        else:
            alpha = lasso_pen_val + ridge_pen_val
            l1_ratio = lasso_pen_val / alpha
            penalty = L1_plus_L2(alpha=alpha, l1_ratio=l1_ratio)

    else:
        if ridge_pen_val is not None:
            raise NotImplementedError

        penalty = WeightedL1(alpha=lasso_pen_val,
                             weights=np.array(lasso_weights).astype(float))

    return penalty
