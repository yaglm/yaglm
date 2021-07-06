from warnings import warn
import numpy as np


def check_no_change(current, prev, tol=None, norm='max'):
    """
    Checks if there is no change between the current current value and the previous value.

    Parameters
    ----------
    current: np.array
        The current value.

    prev: np.array
        The previous value.

    xtol: None, float
        The tolerance stopping criteria i.e. will stop if ||current - prev|| <= tol.

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
        Norm of the difference
    """

    if tol is None:
        return False, None

    if norm == 'max':
        diff_norm = np.abs(current - prev).max()

    elif norm == 'mad':
        diff_norm = np.abs(current - prev).mean()

    elif norm == 'L2':
        diff_norm = np.sqrt(((current - prev) ** 2).sum())

    elif norm == 'rmse':
        diff_norm = np.sqrt(((current - prev) ** 2).mean())

    else:
        raise ValueError("Bad input to norm: {}".format(norm))

    if diff_norm <= tol:
        return True, diff_norm

    else:
        return False, diff_norm


def check_decreasing_loss(current_loss, prev_loss,
                          abs_tol=None, rel_tol=None,
                          on_increase='ignore'):
    """
    Decides whether or not to stop an optimization algorithm if the relative or absolute difference stopping critera are met.

    Parameters
    ----------
    current_loss: float
        The current value of the loss function.

    prev_loss: float, None, np.inf
        The previous value of the loss function. If None or np.inf, will not stop.

    abs_tol: None, float
        The absolute difference tolerance. If None, will not check absolute stopping critera.

    rel_tol: None, float
        The relative difference tolerance. If None, will not check relative stopping critera.

    on_increase: str
        What to do if the loss goes up. Must be one of ['ignore', 'warn', 'error', 'print'].

    Output
    ------
    stop: bool
        True if the loss has stopped decreasing.
    """
    if abs_tol is None and rel_tol is None:
        return False

    if prev_loss is None or prev_loss == np.inf:
        return False

    # maybe check for increasing loss
    if current_loss > prev_loss and on_increase != 'ignore':

        msg = "Increasing loss from {:1.5f} to {:1.5f}".\
            format(prev_loss, current_loss)

        if on_increase == 'error':
            raise ValueError(msg)

        elif on_increase == 'print':
            print(msg)

        elif on_increase == 'warn':
            warn("Increasing loss")

        else:
            raise ValueError("Bad input to on_increase: {}".format(on_increase))

    abs_diff = abs(current_loss - prev_loss)
    # rel_diff = abs_diff / max(abs(current_loss), np.finfo(float).eps)

    # Check absolute stopping condition
    if abs_tol is not None and abs_diff <= abs_tol:
        return True

    if rel_tol is not None and abs_diff <= abs(current_loss) * rel_tol:
        return True
