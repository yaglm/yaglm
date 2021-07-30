import numpy as np


def get_sequence_decr_max(max_val=1, min_val_mult=1e-3, num=20,
                          spacing='lin', decr=True):
    """
    Gets a tuning parameter sequence decreasing from a maximum value.

    Parameters
    ----------
    max_val: float
        The largest value in the sequence.

    min_val_mult: float
        Minimum value = max_val  * min_val_mult

    num: int
        Number of values in the sequence.

    spacing: str
        Determines how points are spaced; must be on of ['lin', 'log'].

    decr: Bool
        Return the sequence in decreasing order.

    Output
    ------
    seq: array-like of floats
        The sequence.

    """
    assert spacing in ['lin', 'log']
    assert min_val_mult <= 1

    min_val = min_val_mult * max_val

    if spacing == 'log':
        assert min_val > 0

    # compute the sequence
    if spacing == 'lin':
        seq = np.linspace(start=min_val,
                          stop=max_val, num=num)

    elif spacing == 'log':
        seq = np.logspace(start=np.log10(min_val),
                          stop=np.log10(max_val),
                          num=num)

    if decr:
        seq = seq[::-1]

    return seq


def get_enet_pen_val_seq(lasso_pen_val_max,
                         pen_vals=None, n_pen_vals=100,
                         pen_min_mult=1e-3, pen_spacing='log',
                         l1_ratio_seq=None, l1_ratio_val=None):
    """
    Sets up the pen_val tuning sequence for eleastic net.
    """
    # only one of these should be not None
    assert sum((l1_ratio_val is None, l1_ratio_seq is None)) <= 2

    # formatting
    if l1_ratio_val is not None:
        tune_l1_ratio = False
        l1_ratio_min = l1_ratio_val
    else:
        tune_l1_ratio = True
        l1_ratio_min = min(l1_ratio_seq)

    if pen_vals is not None:  # user provided pen vals
        pen_vals = np.array(pen_vals)

    else:  # automatically derive tuning sequence

        if l1_ratio_min <= np.finfo(float).eps:
            raise ValueError("Unable to set pen_val_seq using default"
                             "when the l1_ratio is zero."
                             " Either change thel1_ratio, or "
                             "input a sequence of pen_vals yourself!")

        if tune_l1_ratio:
            # setup grid of pen vals for each l1 ratio

            n_l1_ratio_vals = len(l1_ratio_seq)
            pen_vals = np.zeros((n_l1_ratio_vals, n_pen_vals))
            for l1_idx in range(n_l1_ratio_vals):

                # largest pen val for ElasticNet given this l1_ratio
                max_val = lasso_pen_val_max / l1_ratio_seq[l1_idx]

                pen_vals[l1_idx, :] = \
                    get_sequence_decr_max(max_val=max_val,
                                          min_val_mult=pen_min_mult,
                                          num=n_pen_vals,
                                          spacing=pen_spacing)

        else:
            # setup pen val sequence

            max_val = lasso_pen_val_max / l1_ratio_val
            pen_vals = \
                get_sequence_decr_max(max_val=max_val,
                                      min_val_mult=pen_min_mult,
                                      num=n_pen_vals,
                                      spacing=pen_spacing)

    # ensure correct ordering
    if tune_l1_ratio:
        assert pen_vals.ndim == 2
        assert pen_vals.shape[0] == len(l1_ratio_seq)

        # make sure pen vals are always in decreasing order
        for l1_idx in range(pen_vals.shape[0]):
            pen_vals[l1_idx, :] = np.sort(pen_vals[l1_idx, :])[::-1]

    else:
        # make sure pen vals are always in decreasing order
        pen_vals = np.sort(np.array(pen_vals))[::-1]
        pen_vals = pen_vals.reshape(-1)

    return pen_vals


def get_enet_ratio_seq(num=10, min_val=0.1):
    """
    Returns a sequence values for tuning the l1_ratio parameter of ElasticNet.
    As suggested by sklearn.linear_model.ElasticNetCV, we pick values that
    favor larger values of l1_ratio (meaning more lasso).


    In deatil, the sequence is logarithmicly spaced between 1 and min_val.

    Parameters
    ----------
    num: int
        Number of values to return.

    min_val: float
        The smallest value of l1_ratio to return.

    Output
    ------
    values: array-like, shape (num, )
        The values.
    """
    return 1 + min_val - np.logspace(start=0, stop=np.log10(min_val), num=10)
