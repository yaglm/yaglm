from sklearn.utils import check_random_state
from itertools import product

from ya_glm.opt.glm_loss.linear_regression import LinReg, LinRegMultiResp
from ya_glm.opt.glm_loss.logistic_regression import LogReg
from ya_glm.opt.glm_loss.huber_regression import HuberReg, HuberRegMultiResp
from ya_glm.opt.glm_loss.poisson_regression import PoissonReg, PoissonRegMultiResp

from ya_glm.opt.base import check_grad_impl


def run_tests(func, values, step=0.5, atol=1e-2, rtol=0):

    check_grad_impl(func=func, values=values,
                    behavior='error', atol=atol,
                    verbosity=0)


# debugging code
# from scipy.optimize import approx_fprime
# value = values[0]
# func = LinReg(**kws)
# true_grad = approx_fprime(xk=value, f=func.eval, epsilon=1e-4)
# grad = func.grad(value)
# print(kws['fit_intercept'])
# print(np.linalg.norm(grad - true_grad))


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


def test_lin_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        run_tests(func=LinReg(**kws),
                  values=values)


def test_log_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        kws['y'] = (kws['y'] > 0).astype(float)

        run_tests(func=LogReg(**kws),
                  values=values)


def test_huber_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):
        for knot in [.1, 1]:
            kws['loss_kws'] = {'knot': knot}

            run_tests(func=HuberReg(**kws),
                      values=values)


def test_poisson_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        kws['y'] = abs(kws['y'])

        run_tests(func=PoissonReg(**kws),
                  values=values)
