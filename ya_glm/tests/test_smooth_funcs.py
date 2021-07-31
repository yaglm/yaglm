
from ya_glm.opt.glm_loss.linear_regression import LinReg
from ya_glm.opt.glm_loss.logistic_regression import LogReg
from ya_glm.opt.glm_loss.huber_regression import HuberReg
from ya_glm.opt.glm_loss.poisson_regression import PoissonReg

from ya_glm.tests.opt_checking_utils import check_smooth, value_func_gen


# debugging code
# from scipy.optimize import approx_fprime
# value = values[0]
# func = LinReg(**kws)
# true_grad = approx_fprime(xk=value, f=func.eval, epsilon=1e-4)
# grad = func.grad(value)
# print(kws['fit_intercept'])
# print(np.linalg.norm(grad - true_grad))

def test_lin_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        check_smooth(func=LinReg(**kws),
                     values=values)


def test_log_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        kws['y'] = (kws['y'] > 0).astype(float)

        check_smooth(func=LogReg(**kws),
                     values=values)


def test_huber_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):
        for knot in [.1, 1]:
            kws['knot'] = knot

            check_smooth(func=HuberReg(**kws),
                         values=values)


def test_poisson_reg():
    for idx, (kws, values) in enumerate(value_func_gen(3)):

        kws['y'] = abs(kws['y'])

        check_smooth(func=PoissonReg(**kws),
                     values=values)
