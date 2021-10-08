import numpy as np

from yaglm.opt.base import Func
from yaglm.opt.utils import decat_coef_inter_mat
from yaglm.opt.utils import decat_coef_inter_vec


class WithIntercept(Func):
    def __init__(self, func):
        self.func = func

    def _eval(self, x):
        coef, _ = decat_coef_inter_vec(x)
        return self.func.eval(coef)

    def _grad(self, x):
        coef, _ = decat_coef_inter_vec(x)
        g = self.func.grad(coef)
        return np.concatenate([[0], g])

    def _prox(self, x, step):
        coef, intercept = decat_coef_inter_vec(x)
        p = self.func.prox(coef, step)
        return np.concatenate([[intercept], p])  # TODO: check

    @property
    def grad_lip(self):
        return self.func.grad_lip

    @property
    def is_smooth(self):
        return self.func.is_smooth

    @property
    def is_proximable(self):
        return self.func.is_proximable


class MatWithIntercept(Func):
    def __init__(self, func):
        self.func = func

    def eval(self, x):
        coef, _ = decat_coef_inter_mat(x)
        return self.func.eval(coef)

    def grad(self, x):
        coef, _ = decat_coef_inter_mat(x)
        g = self.func.grad(coef)
        g = np.vstack([np.zeros(x.shape[1]), g])
        return g

    def prox(self, x, step=1):
        coef, intercept = decat_coef_inter_mat(x)
        p = self.func.prox(coef, step)
        p = np.vstack([intercept, p])
        return p

    @property
    def grad_lip(self):
        return self.func.grad_lip

    @property
    def is_smooth(self):
        return self.func.is_smooth

    @property
    def is_proximable(self):
        return self.func.is_proximable
