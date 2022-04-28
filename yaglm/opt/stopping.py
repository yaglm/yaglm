from warnings import warn
import numpy as np


def check_no_change(current, prev, norm='max', tol=None,
                    rel_crit=False, tol_eps=np.finfo(float).eps):
    """
    Checks if there is no change between the current current value and the previous value.

    Parameters
    ----------
    current: np.array
        The current value.

    prev: np.array
        The previous value.

    tol: None, float
        The tolerance stopping criteria i.e. will stop if ||current - prev|| <= tol. If None, will always return stop=False.

    rel_crit: float
        Should the tolerance be computed on a relative scale i.e. ||current - prev|| / (||prev|| + tol_eps) <= tol

    tol_eps: float
        Epsilon value for relative tolerance.

    norm: str
        What norm to use. Must be one of ['max', 'mad', 'L2', 'rmse']

        'max': L_infty norm

        'mad': mean absolute error

        'L2': euclidean norm

        'rmse': root mean square error

    Output
    ------
    stop, diff_norm

    stop: bool
        Whether or not to stop.

    diff_norm: float
        Norm of the difference. If rel_crit=True this is the relative difference.
    """
    if tol is None:
        return False, None

    # define norm
    if norm == 'max':
        def f(x): return np.abs(x).max()

    elif norm == 'mad':
        def f(x): return np.abs(x).mean()

    elif norm == 'L2':
        def f(x): return np.sqrt(((x) ** 2).sum())

    elif norm == 'rmse':
        def f(x): return np.sqrt(((x) ** 2).mean())

    else:
        raise ValueError("Bad input to norm: {}".format(norm))

    # compute norm of difference
    diff_norm = f(current - prev)
    if rel_crit:
        diff_norm /= (f(prev) + tol_eps)

    # check stopping criteria
    if diff_norm <= tol:
        return True, diff_norm

    else:
        return False, diff_norm


def check_decreasing_loss(current, prev, tol=None,
                          rel_crit=False, tol_eps=np.finfo(float).eps,
                          on_increase='ignore'):
    """
    Decides whether or not to stop an optimization algorithm if the relative or absolute difference stopping critera are met.

    Parameters
    ----------
    current_loss: float
        The current value of the loss function.

    prev_loss: float, None, np.inf
        The previous value of the loss function. If None or np.inf, will not stop.

    tol: None, float
        The tolerance stopping criteria i.e. will stop if current - prev <= tol. If None, will always return stop=False.

    rel_crit: float
        Should the tolerance be computed on a relative scale i.e. current - prev / (|prev| + tol_eps) <= tol

    tol_eps: float
        Epsilon value for relative tolerance.

    on_increase: str
        What to do if the loss goes up. Must be one of ['ignore', 'warn', 'error', 'print'].

    Output
    ------
    stop: bool
        True if the loss has stopped decreasing.
    """
    if tol is None:
        return False

    # maybe check for increasing loss
    if current > prev and on_increase != 'ignore':

        msg = "Increasing loss from {:1.5f} to {:1.5f}".\
            format(prev, current)

        if on_increase == 'error':
            raise ValueError(msg)

        elif on_increase == 'print':
            print(msg)

        elif on_increase == 'warn':
            warn("Increasing loss")

        else:
            raise ValueError("Bad input to on_increase: {}".format(on_increase))

    # compute difference
    diff = abs(current - prev)
    if rel_crit:
        diff /= (abs(prev) + tol_eps)

    # check stopping criterla
    if diff <= tol:
        return True
    else:
        return False
