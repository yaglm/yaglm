import numpy as np
from textwrap import dedent

from ya_glm.opt.base import Func
from ya_glm.opt.utils import sign_never_0, safe_vectorize

########
# SCAD #
########


def scad_eval_1d(x, pen_val, a=3.7):
    """
    Evaluates the SCAD penalty funciton.

    Parameters
    ----------
    x: float

    a: float

    pen_val: float

    Output
    ------
    value: float
    """
    # https://statisticaloddsandends.wordpress.com/2018/07/31/the-scad-penalty/

    abs_x = abs(x)
    if abs_x <= pen_val:
        return pen_val * abs_x

    elif abs_x <= a * pen_val:
        return (2 * a * pen_val * abs_x - x ** 2 - pen_val ** 2) \
            / (2 * (a - 1))

    else:
        return 0.5 * (a + 1) * pen_val ** 2


_vec_scad_eval = safe_vectorize(scad_eval_1d)


def scad_eval(x, pen_val, a=3.7):
    return _vec_scad_eval(x, pen_val, a).sum()


def scad_grad_1d(x, pen_val, a=3.7):
    """
    Evaluates the SCAD gradient.

    Parameters
    ----------
    x: float

    a: float

    pen_val: float

    Output
    ------
    value: float
    """
    abs_x = abs(x)
    if abs_x <= pen_val:
        return sign_never_0(x) * pen_val

    elif abs_x <= a * pen_val:
        return np.sign(x) * (a * pen_val - abs_x) / (a - 1)

    else:
        return 0


scad_grad = safe_vectorize(scad_grad_1d)


def scad_prox_1d(x, pen_val, a=3.7):
    """
    Evaluates the proximal operator of the SCAD function

    Parameters
    ----------
    x: float

    a: float

    pen_val: float

    Output
    ------
    value: float
    """
    abs_x = abs(x)
    if abs_x <= 2 * pen_val:
        return np.sign(x) * np.max([0, abs_x - pen_val])

    elif abs_x <= a * pen_val:
        return ((a - 1) * x - np.sign(x) * a * pen_val) / (a - 2)

    else:
        return x


scad_prox = safe_vectorize(scad_prox_1d)


class SCAD(Func):
    def __init__(self, pen_val=1, a=3.7):
        self.pen_val = pen_val
        self.a = a
    
    def _eval(self, x):
        return scad_eval(x, pen_val=self.pen_val, a=self.a)
    
    def _grad(self, x):
        return scad_grad(x, pen_val=self.pen_val, a=self.a)

#######
# MCP #
#######


def mcp_eval_1d(x, pen_val, a=2):
    """
    Evaluates the MCP penalty funciton at |x|.

    Parameters
    ----------
    x: float

    pen_val: float

    a: float

    Output
    ------
    value: float
    """
    # https://statisticaloddsandends.wordpress.com/2019/12/09/the-minimax-concave-penalty-mcp/
    abs_x = abs(x)
    if abs_x <= a * pen_val:
        return pen_val * abs_x - x ** 2 / (2 * a)

    else:
        return 0.5 * a * pen_val ** 2


_vec_mcp_eval = safe_vectorize(mcp_eval_1d)


def mcp_eval(x, pen_val, a=2):
    return _vec_mcp_eval(x, pen_val, a).sum()


def mcp_grad_1d(x, a, pen_val):
    """
    Gradient of the MCP function.

    Parameters
    ----------
    x: float

    pen_val: float

    a: float

    Output
    ------
    value: float
    """
    abs_x = abs(x)
    if abs_x <= a * pen_val:
        return sign_never_0(x) * (pen_val - abs_x / a)

    else:
        return 0


mcp_grad = safe_vectorize(mcp_grad_1d)


def mcp_prox_1d(x, pen_val, a=2):
    """
    Proximal operator of the MCP function.

    Parameters
    ----------
    x: float

    pen_val: float

    a: float

    Output
    ------
    value: float
    """

    abs_x = abs(x)
    if abs_x <= pen_val:
        return 0

    elif abs_x <= a * pen_val:
        return np.sign(x) * (abs_x - pen_val) / (1 - (1 / a))

    else:
        return x


mcp_prox = safe_vectorize(mcp_prox_1d)


class MCP(Func):

    def __init__(self, pen_val=1, a=2):
        self.pen_val = pen_val
        self.a = a

    def _eval(self, x):
        return mcp_eval(x=x, pen_val=self.pen_val, a=self.a)

    def _grad(self, x):
        return mcp_grad(x=x, pen_val=self.pen_val, a=self.a)


###############
# Inverse Log #
###############


class InvLog(Func):
    """
    g(x) = pen_val * log(epsilon + |x|)
    """
    def __init__(self, pen_val=1, epsilon=0):
        self.pen_val = pen_val
        self.epsilon = epsilon

    def _eval(self, x):
        return self.pen_val * np.log(self.epsilon + np.abs(x))

    def _grad(self, x):
        return self.pen_val / (self.epsilon + np.abs(x))

######
# Lq #
######


class Lq(Func):
    """
    g(x) = pen_val * |x|^q
    """
    def __init__(self, pen_val=1, q=0.5):
        self.pen_val = pen_val
        self.q = q

    def _eval(self, x):
        return self.pen_val * np.abs(x) ** self.q

    def _grad(self, x):
        return self.pen_val * self.q * np.abs(x) ** (self.q - 1)


_pen_fuc_docs = dict(
    options=dedent("""
    pen_func_kws: str
        Which folded concave penalty function to use. Must be one of ['SCAD', 'MCP', 'InvLog', 'Lq', 'Lasso'].

    pen_kws: dict
        Optional key-word arguments for folded concave penalty.
        """),

    pen_param=dedent("""
    pen_val: float
        Penalty parameter value for the folded concave penalty.
        """)
)


def get_penalty_func(pen_func, pen_val, pen_func_kws={}):
    pen_func = pen_func.lower()
    assert pen_func in ['scad', 'mcp', 'invlog', 'lq', 'lasso']

    if pen_func == 'scad':
        c = SCAD

    elif pen_func == 'mcp':
        c = MCP

    elif pen_func == 'lq':
        c = Lq

    elif pen_func == 'invlog':
        c = InvLog

    return c(pen_val=pen_val, **pen_func_kws)


get_penalty_func.__doc__ = dedent("""
Gets a folded concave penalty function object.

Parameters
----------
{options}
{pen_param}

Output
------
penaly: FoldedPenalty
""".format(**_pen_fuc_docs))
