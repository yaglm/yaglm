import numpy as np

from yaglm.opt.utils import sign_never_0, safe_vectorize

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


# def scad_prox_1d(x, pen_val, a=3.7):
#     """
#     Evaluates the proximal operator of the SCAD function

#     Parameters
#     ----------
#     x: float

#     a: float

#     pen_val: float

#     Output
#     ------
#     value: float
#     """
#     abs_x = abs(x)
#     if abs_x <= 2 * pen_val:
#         return np.sign(x) * np.max([0, abs_x - pen_val])

#     elif abs_x <= a * pen_val:
#         return ((a - 1) * x - np.sign(x) * a * pen_val) / (a - 2)

#     else:
#         return x

def scad_prox_1d_with_step(x, pen_val, a=3.7, step=1):
    """
    Evaluates the proximal operator of the SCAD function

    Parameters
    ----------
    x: float

    a: float

    pen_val: float

    step: float

    Output
    ------
    value: float
    """
    # TODO: clean this up
    abs_x = abs(x)

    sol_1 = max(0, abs_x - step * pen_val)
    sol_2 = ((a - 1) * abs_x - step * pen_val * a) / (a - 1 - step)
    sol_2 = abs(sol_2)
    sol_3 = abs_x

    prox_obj_1 = scad_eval_1d(x=sol_1, pen_val=pen_val, a=a) + \
        (0.5 / step) * (sol_1 - abs_x) ** 2

    prox_obj_2 = scad_eval_1d(x=sol_2, pen_val=pen_val, a=a) + \
        (0.5 / step) * (sol_2 - abs_x) ** 2

    prox_obj_3 = scad_eval_1d(x=sol_3, pen_val=pen_val, a=a) + \
        (0.5 / step) * (sol_3 - abs_x) ** 2

    objs = [prox_obj_1, prox_obj_2, prox_obj_3]
    idx_min = np.argmin(objs)

    if idx_min == 0:
        return np.sign(x) * sol_1

    elif idx_min == 1:
        return np.sign(x) * sol_2

    elif idx_min == 2:
        return np.sign(x) * sol_3


scad_prox = safe_vectorize(scad_prox_1d_with_step)


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


# def mcp_prox_1d(x, pen_val, a=2):
#     """
#     Proximal operator of the MCP function.

#     Parameters
#     ----------
#     x: float

#     pen_val: float

#     a: float

#     Output
#     ------
#     value: float
#     """

#     abs_x = abs(x)
#     if abs_x <= pen_val:
#         return 0

#     elif abs_x <= a * pen_val:
#         return np.sign(x) * (abs_x - pen_val) / (1 - (1 / a))

#     else:
#         return x


def mcp_prox_1d_with_step(x, pen_val, a=2, step=1):
    """
    Proximal operator of the MCP function.

    Parameters
    ----------
    x: float

    pen_val: float

    a: float

    step: float

    Output
    ------
    value: float
    """
    # TODO: clean this up

    abs_x = abs(x)

    sol_1 = (1 / (1 - (step / a))) * max(0, abs_x - step * pen_val)
    sol_2 = abs_x

    prox_obj_1 = mcp_eval_1d(x=sol_1, pen_val=pen_val, a=a) + \
        (0.5 / step) * (sol_1 - abs_x) ** 2

    prox_obj_2 = mcp_eval_1d(x=sol_2, pen_val=pen_val, a=a) + \
        (0.5 / step) * (sol_2 - abs_x) ** 2

    objs = [prox_obj_1, prox_obj_2]
    idx_min = np.argmin(objs)

    if idx_min == 0:
        return np.sign(x) * sol_1

    elif idx_min == 1:
        return np.sign(x) * sol_2


mcp_prox = safe_vectorize(mcp_prox_1d_with_step)
