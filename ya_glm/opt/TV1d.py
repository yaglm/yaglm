import numpy as np

from lightning.impl.prox_fast import prox_tv1d
from ya_glm.opt.base import Func


class TV1d(Func):

    def __init__(self, mult=1):
        self.mult = mult

    def _prox(self, x, step=1):
        p = x.copy().astype(float)
        # TODO: double check formula
        # in place!
        prox_tv1d(w=p, stepsize=self.mult * step)
        return p

    def _eval(self, x):
        # https://github.com/scikit-learn-contrib/lightning/blob/master/lightning/impl/penalty.py
        return self.mult * np.sum(np.abs(np.diff(x)))
