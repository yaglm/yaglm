import numpy as np
from sklearn.utils import check_random_state
from itertools import product

from ya_glm.opt.linear_regression import LinRegLoss
from ya_glm.opt.logistic_regression import LogRegLoss
from ya_glm.opt.base import check_grad_impl


def run_tests(func, values, step=0.5, atol=1e-2, rtol=0):

    check_grad_impl(func=func, values=values,
                    behavior='error', atol=atol,
                    verbosity=0)


def value_func_gen():

    rng = check_random_state(234)

    sizes = [(10, 20), (20, 10), (10, 10)]
    for size, fit_intercept in product(sizes, [True, False]):
        X = rng.normal(size=size)
        y = rng.normal(size=size[0])

        if fit_intercept:
            value = [rng.normal(size=size[1] + 1)]
        else:
            value = [rng.normal(size=size[1])]

        yield {'X': X, 'y': y, 'fit_intercept': fit_intercept}, value


def test_lin_reg():
    for idx, (kws, values) in enumerate(value_func_gen()):

        run_tests(func=LinRegLoss(**kws),
                  values=values)


def test_log_reg():
    for idx, (kws, values) in enumerate(value_func_gen()):

        kws['y'] = (kws['y'] > 0).astype(float)

        run_tests(func=LogRegLoss(**kws),
                  values=values)
