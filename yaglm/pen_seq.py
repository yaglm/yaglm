import numpy as np


def get_sequence_decr_max(max_val=1, min_val_mult=1e-3, num=20,
                          spacing='log', decr=True):
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


def get_mix_val_seq(num=10,
                    min_val=1e-2,
                    spacing='log',
                    prefer_larger=True):
    """
    Sets up the mix_val tuning sequence for a mixed penalty (e.g. ElasticNet) that looks like

    pen_val * mix_val * primary_penalty(coef) + pen_val * (1 - mix_val) * second_penalty(coef)

    Note mix_val takes values in [0, 1].

    Parameters
    ----------
    num: int
        Number of mix_vals in the automatically created mix_val tune sequence.

    min_val: float
        Smallest value of the mixing parameter.

    spacing: str
        Determines the spacing of the values. Must be one of ['lin', 'log'].

    prefer_larger: bool
        Prefer values closer to 1 for log spacing e.g. as suggested by sklearn.linear_model.ElasticNetCV.

    Output
    ------
    values: array-like, shape (num, )
        The values.
    """
    if spacing == 'lin':
        values = np.linspace(start=min_val, stop=1, num=num)

    elif spacing == 'log':

        if prefer_larger:
            values = 1 + min_val - np.logspace(start=0, stop=np.log10(min_val),
                                               num=num)
        else:
            values = np.logspace(start=np.log10(min_val), stop=0,
                                 num=num)

    return np.sort(values)
