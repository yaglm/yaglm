from sklearn.utils import check_random_state
from itertools import product
import numpy as np

from ya_glm.opt.base import check_grad_impl, check_prox_impl


def check_smooth(func, values, step=0.5, atol=1e-2, rtol=0):

    check_grad_impl(func=func, values=values,
                    behavior='error', atol=atol,
                    verbosity=0)


def value_func_gen(n_values=1):

    rng = check_random_state(234)

    sizes = [(10, 20), (20, 10), (10, 10)]
    for size, fit_intercept in product(sizes, [True, False]):
        X = rng.normal(size=size)
        y = rng.normal(size=size[0])

        if fit_intercept:
            value = [rng.normal(size=size[1] + 1)
                     for _ in range(n_values)]
        else:
            value = [rng.normal(size=size[1])
                     for _ in range(n_values)]

        yield {'X': X, 'y': y, 'fit_intercept': fit_intercept}, value


def check_nonsmooth(func, values, step=0.5, atol=1e-2, rtol=0):

    # opt_kws = {'options': {'disp': True}}

    check_prox_impl(func=func, values=values, step=step,
                    behavior='error', rtol=rtol, atol=atol,
                    verbosity=0)


def check_smooth_with_prox(func, values, step=0.5, atol=1e-2, rtol=0):

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

    yield {'mult': 3}, values_with_scalar

    yield {'mult': 3}, values_all_vec

    yield {'mult': 3, 'weights': np.arange(10)}, values_all_vec


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
