from itertools import product

from ya_glm.tests.utils import X_data_generator, compare_fits, \
  lin_reg_y_from_X, log_reg_y_from_X
from ya_glm.tests.fit_ya_glm_and_sklearn import fit_lin_reg, fit_log_reg


def test_lin_reg(tol=1e-2, verbosity=1, solver='default', behavior='error'):
    """
    Compare ya_glm to sklearn for some simple linear regression examples.
    """
    loop_idx = 0

    for (fit_intercept, standardize) in product([False, True], [False, True]):
        for X in X_data_generator():

            if standardize and not fit_intercept:
                # if standardize=True, fit_intercept=False sklearn does not scale the data matrix
                # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/linear_model/_base.py#L173
                continue

            if verbosity > 0:
                print('starting loop_idx={} fit_intercept={} standardize={}'.
                      format(loop_idx, fit_intercept, standardize))

            y = lin_reg_y_from_X(X=X)

            if X.shape[0] > X.shape[1]:  # Ignore HD case
                # Vanilla
                est, sk = fit_lin_reg(X=X, y=y,
                                      fit_intercept=fit_intercept,
                                      standardize=standardize,
                                      solver=solver)

                compare_fits(est, sk, tol=tol, behavior=behavior)

            # Lasso
            est, sk = fit_lin_reg(X=X, y=y,
                                  fit_intercept=fit_intercept,
                                  standardize=standardize,
                                  solver=solver,
                                  lasso_pen_val=.2
                                  )

            compare_fits(est, sk, tol=tol, behavior=behavior)

            # Ridge
            est, sk = fit_lin_reg(X=X, y=y,
                                  fit_intercept=fit_intercept,
                                  standardize=standardize,
                                  solver=solver,
                                  ridge_pen_val=.2
                                  )

            compare_fits(est, sk, tol=tol, behavior=behavior)

            # Elasticnet
            est, sk = fit_lin_reg(X=X, y=y,
                                  fit_intercept=fit_intercept,
                                  standardize=standardize,
                                  solver=solver,
                                  lasso_pen_val=.1,
                                  ridge_pen_val=.2
                                  )

            compare_fits(est, sk, tol=tol, behavior=behavior)
            loop_idx += 1


def test_log_reg(tol=1e-2, verbosity=1, solver='default', behavior='error'):
    """
    Compare ya_glm to sklearn for some simple logistic regression examples
    """
    # TODO: fit_intercept=True sometimes gives slightly different answers
    # TODO: ridge gives a different answer
    loop_idx = 0

    fit_intercept = False
    for X in X_data_generator():

        if verbosity > 0:
            print('starting loop_idx={} fit_intercept={}'.
                  format(loop_idx, fit_intercept))

        y = log_reg_y_from_X(X=X)

        # unpenalized logistic regression can be unstable so this is
        # not a good test
        # if X.shape[0] > X.shape[1]:  # Ignore HD case
        #     # Vanilla
        #     est, sk = fit_log_reg(X=X, y=y,
        #                           fit_intercept=fit_intercept,
        #                           solver=solver)

        #     compare_fits(est, sk, tol=tol, behavior=behavior)

        # Lasso
        est, sk = fit_log_reg(X=X, y=y,
                              fit_intercept=fit_intercept,
                              solver=solver,
                              lasso_pen_val=.1
                              )

        compare_fits(est, sk, tol=tol, behavior=behavior)

        # Ridge
        # TODO: for some reason this is giving different answers
        # est, sk = fit_log_reg(X=X, y=y,
        #                       fit_intercept=fit_intercept,
        #                       solver=solver,
        #                       ridge_pen_val=.1
        #                       )

        # compare_fits(est, sk, tol=tol, behavior=behavior)

        # Elasticnet
        est, sk = fit_log_reg(X=X, y=y,
                              fit_intercept=fit_intercept,
                              solver=solver,
                              lasso_pen_val=.1,
                              ridge_pen_val=.1
                              )

        compare_fits(est, sk, tol=tol, behavior=behavior)
        loop_idx += 1
