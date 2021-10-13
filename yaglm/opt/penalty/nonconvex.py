from yaglm.autoassign import autoassign
from yaglm.opt.base import EntrywiseFunc
from yaglm.opt.nonconvex_utils import scad_eval, scad_grad, scad_prox, \
    mcp_eval, mcp_grad, mcp_prox


class SCAD(EntrywiseFunc):

    @autoassign
    def __init__(self, pen_val=1, a=3.7): pass

    def _eval(self, x):
        return scad_eval(x, pen_val=self.pen_val, a=self.a)

    def _grad(self, x):
        return scad_grad(x, pen_val=self.pen_val, a=self.a)

    def _prox(self, x, step=1):
        return scad_prox(x, pen_val=self.pen_val, a=self.a, step=step)

    @property
    def is_smooth(self):
        return False

    @property
    def is_proximable(self):
        return True

    @property
    def fcp_data(self):
        """
        See Definition 3.1 of https://arxiv.org/pdf/2107.03494.pdf
        """
        return {'a0': 1, 'a1': 1, 'b1': 1, 'b2': self.a}


class MCP(EntrywiseFunc):

    @autoassign
    def __init__(self, pen_val=1, a=2): pass

    def _eval(self, x):
        return mcp_eval(x=x, pen_val=self.pen_val, a=self.a)

    def _grad(self, x):
        return mcp_grad(x=x, pen_val=self.pen_val, a=self.a)

    def _prox(self, x, step=1):
        return mcp_prox(x=x, pen_val=self.pen_val, a=self.a, step=step)

    @property
    def is_smooth(self):
        return False

    @property
    def is_proximable(self):
        return True

    @property
    def fcp_data(self):
        """
        See Definition 3.1 of https://arxiv.org/pdf/2107.03494.pdf
        """
        return {'a0': 1, 'a1': 1 - (1 / self.a), 'b1': 1, 'b2': self.a}

# class Log(EntrywiseFunc):
#     """
#     g(x) = pen_val * log(perturb + |x|)
#     """

#     @autoassign
#     def __init__(self, pen_val=1, perturb=1e-2): pass

#     def _eval(self, x):
#         return self.pen_val * np.log(self.perturb + np.abs(x))

#     def _grad(self, x):
#         return self.pen_val / (self.perturb + np.abs(x))

#     @property
#     def is_smooth(self):
#         return False


# for prox see p19 https://arxiv.org/pdf/1502.03175.pdf
# class Lq(EntrywiseFunc):
#     """
#     g(x) = pen_val * |x|^q
#     """
#     @autoassign
#     def __init__(self, pen_val=1, q=0.5): pass

#     def _eval(self, x):
#         return self.pen_val * np.abs(x) ** self.q

#     def _grad(self, x):
#         return self.pen_val * self.q * np.abs(x) ** (self.q - 1)

#     @property
#     def is_smooth(self):
#         return self.q > 1

avail_nonconvex = ['sacd', 'mcp']


def get_nonconvex_func(name, pen_val, second_param=None):

    kws = {'pen_val': pen_val}

    if name.lower() == 'scad':
        if second_param is not None:
            kws['a'] = second_param
        return SCAD(**kws)

    elif name.lower() == 'mcp':
        if second_param is not None:
            kws['a'] = second_param
        return MCP(**kws)

    # elif name.lower() == 'log':
    #     if second_param is not None:
    #         kws['perturb'] = second_param
    #     return Log(**kws)

    # elif name.lower() == 'lq':
    #     if second_param is not None:
    #         kws['q'] = second_param
    #     return Lq(**kws)

    else:
        raise ValueError("{} is not a valid input")
