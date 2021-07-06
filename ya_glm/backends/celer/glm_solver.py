from time import time
import numpy as np

from andersoncd.weighted_lasso import celer_primal_path as lin_reg_path


def solve_glm(X, y,
              loss_func='lin_reg',
              fit_intercept=True,

              lasso_pen=None,
              lasso_weights=None,
              groups=None,
              L1to2=False,
              nuc=False,
              L2_pen=None,
              L2_weights=None,
              tikhonov=None,

              coef_init=None,
              intercept_init=None,
              max_iter=20, max_epochs=50000,
              p0=10, verbose=0, tol=1e-4, prune=0,
              return_n_iter=False):

    #############################
    # check what is implemented #
    #############################
    if loss_func == 'lin_reg':
        path = lin_reg_path
    else:
        raise NotImplementedError("{} is not supported".format(loss_func))

    if groups is not None:
        raise NotImplementedError("gruops is not supported")

    if L1to2:
        raise NotImplementedError("L1to2 is not yet supported")

    if nuc:
        raise NotImplementedError("nuc is not yet supported")

    if tikhonov is not None:
        raise NotImplementedError("tikhonov is not yet supported")

    if L2_pen is not None or L2_weights is not None:
        raise NotImplementedError("Ridge penalty not yet supported")

    ##############
    # formatting #
    ##############

    if lasso_pen is None and lasso_weights is not None:
        lasso_pen = 1
    elif lasso_pen is None:
        lasso_pen = 0

    # if L2_pen is None and L2_weights is not None:
    #     L2_pen = 1
    # elif L2_pen is None:
    #     L2_pen = 0

    if coef_init is not None:
        coef_init = coef_init.astype(X.dtype)

    if fit_intercept:
        m = X.mean(axis=0)
    else:
        m = 0

    start_time = time()
    _, coef, dual_gap, n_iters = path(X=X - m, y=y,
                                      eps=None, n_alphas=None,
                                      alphas=[lasso_pen],
                                      coef_init=coef_init,
                                      max_iter=max_iter,
                                      max_epochs=max_epochs,
                                      p0=p0,
                                      verbose=verbose,
                                      tol=tol,
                                      prune=prune,
                                      weights=lasso_weights,
                                      return_n_iter=True)

    coef = coef.reshape(-1)
    opt_data = {'dual_gap': dual_gap,
                'runtime': time() - start_time,
                'n_iters': n_iters}

    if fit_intercept:
        # TODO: double chekc
        intercept = y.mean() - m.T @ coef
    else:
        intercept = None

    return coef, intercept, opt_data


def solve_glm_path(X, y,
                   lasso_pen_seq=None,
                   L2_pen_seq=None,
                   check_decr=True,
                   loss_func='lin_reg',
                   fit_intercept=True,

                   lasso_weights=None,
                   groups=None,
                   L1to2=False,
                   nuc=False,
                   L2_pen=None,
                   L2_weights=None,
                   tikhonov=None,

                   coef_init=None,
                   intercept_init=None,
                   max_iter=20, max_epochs=50000,
                   p0=10, verbose=0, tol=1e-4, prune=0,
                   return_n_iter=False):

    #############################
    # check what is implemented #
    #############################

    if loss_func == 'lin_reg':
        path = lin_reg_path
    else:
        raise NotImplementedError("{} is not supported".format(loss_func))

    if groups is not None:
        raise NotImplementedError("gruops is not supported")

    if L1to2:
        raise NotImplementedError("L1to2 is not yet supported")

    if nuc:
        raise NotImplementedError("nuc is not yet supported")

    if tikhonov is not None:
        raise NotImplementedError("tikhonov is not yet supported")

    if L2_pen_seq is not None:
        raise NotImplementedError

    ##############
    # Formatting #
    ##############
    if check_decr:
        if lasso_pen_seq is not None:
            assert all(np.diff(lasso_pen_seq) <= 0)

        if L2_pen_seq is not None:
            assert all(np.diff(L2_pen_seq) <= 0)

    if coef_init is not None:
        coef_init = coef_init.astype(X.dtype)

    if fit_intercept:
        m = X.mean(axis=0)
    else:
        m = 0

    # Solve the path!
    _, coefs, dual_gaps, n_iters = path(X=X - m, y=y,
                                        eps=None, n_alphas=None,
                                        alphas=lasso_pen_seq,
                                        coef_init=coef_init,
                                        max_iter=max_iter,
                                        max_epochs=max_epochs,
                                        p0=p0,
                                        verbose=verbose,
                                        tol=tol,
                                        prune=prune,
                                        weights=lasso_weights,
                                        return_n_iter=True)

    # format path output
    out = []
    for idx in range(coefs.shape[1]):

        opt_data = {'dual_gap': dual_gaps[idx], 'n_iters': n_iters[idx]}
        param = {'lasso_pen': lasso_pen_seq[idx]}

        coef = coefs[:, idx]

        if fit_intercept:
            # TODO: double check
            intercept = y.mean() - m.T @ coef
        else:
            intercept = None

        fit_out = {'coef': coef,
                   'intercept': intercept,
                   'opt_data': opt_data}

        out.append((fit_out, param))

    return out
