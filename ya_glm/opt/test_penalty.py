import numpy as np

from ya_glm.opt.penalty import LassoPenalty, RidgePenalty,\
    LassoRidgePenalty
from ya_glm.opt.base import check_grad_impl, check_prox_impl

# TODO: need to adjust these tests with new intercept handling
def run_tests_nonsmooth(func, values, step=0.5, atol=1e-2, rtol=0):

    # opt_kws = {'options': {'disp': True}}

    check_prox_impl(func=func, values=values, step=step,
                    behavior='error', rtol=rtol, atol=atol,
                    verbosity=0)


def run_tests_smooth(func, values, step=0.5, atol=1e-2, rtol=0):

    check_grad_impl(func=func, values=values,
                    behavior='error', atol=atol,
                    verbosity=0)

    check_prox_impl(func=func, values=values, step=step,
                    behavior='error', rtol=rtol, atol=atol,
                    verbosity=0)


def penalty_value_func_gen():

    values_all_vec = [np.zeros(10),
                      np.ones(10),
                      -np.ones(10),
                      100 * np.ones(10),
                      -100 * np.ones(10),
                      0.1 * np.ones(10),
                      -0.1 * np.ones(10)]

    values_with_scalar = [0, 1, -1] + values_all_vec

    yield {'mult': 3, 'fit_intercept': False}, values_with_scalar

    yield {'mult': 3, 'fit_intercept': True}, values_all_vec

    yield {'mult': 3, 'fit_intercept': False, 'weights': np.arange(10)}, values_all_vec

    yield {'mult': 3, 'fit_intercept': True, 'weights': np.arange(9)}, values_all_vec


def penalty_value_func_get_l1_l2():
    for kws, values in penalty_value_func_gen():
        if 'weights' in kws:
            kws['lasso_weights'] = kws['weights']
            kws['ridge_weights'] = kws['weights']
            del kws['weights']

        if 'mult' in kws:
            kws['lasso_mult'] = kws['mult']
            kws['ridge_mult'] = kws['mult']
            del kws['mult']

        yield kws, values


def test_lasso():
    for idx, (kws, values) in enumerate(penalty_value_func_gen()):
        run_tests_nonsmooth(func=LassoPenalty(**kws),
                            values=values)


def test_ridge():
    for idx, (kws, values) in enumerate(penalty_value_func_gen()):
        run_tests_smooth(func=RidgePenalty(**kws),
                         values=values)


def test_lasso_ridge():
    for idx, (kws, values) in enumerate(penalty_value_func_get_l1_l2()):
        run_tests_nonsmooth(func=LassoRidgePenalty(**kws),
                            values=values)
