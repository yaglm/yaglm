import numpy as np
from numbers import Number


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


def get_pen_val_seq(n_pen_vals=100,
                    pen_vals=None,
                    pen_val_max=None,
                    pen_min_mult=1e-3,
                    pen_spacing='log'):
    """
    Returns the penalty value sequence in decreaseing order.

    Parameters
    ----------
    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence.

    pen_vals: None, array-like
        (Optional) User provided penalty value sequence.

    pen_val_max: None, float
        The largest reasonable penalty value. Required to automatically created penalty sequence.

    pen_min_mult: float
        Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    Output
    ------
    pen_val_seq: array-like
        Penalty value sequence in decreaseing order.

    """
    if pen_vals is None:
        assert pen_val_max is not None

        pen_val_seq = \
            get_sequence_decr_max(max_val=pen_val_max,
                                  min_val_mult=pen_min_mult,
                                  num=n_pen_vals,
                                  spacing=pen_spacing)

    else:
        pen_val_seq = np.array(pen_vals)

    # ensure decreasing
    pen_val_seq = np.sort(pen_val_seq)[::-1]
    return pen_val_seq


def get_mix_param_seq(mix_vals=None,
                      n_mix_vals=10,
                      min_val=1e-2,
                      spacing='log',
                      prefer_larger=True):

    """
    Sets up the mix_val tuning sequence for a mixed penalty (e.g. ElasticNet) that looks like

    pen_val * mix_val * primary_penalty(coef) + pen_val * (1 - mix_val) * second_penalty(coef)

    Note mix_val takes values in [0, 1].

    Parameters
    ----------
    mix_vals: None, float, array-like
        (Optional) User specified sequence of mix vals. If None, will automatically create a sequence of mix vals.

    n_mix_vals: int
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
    if mix_vals is not None:
        values = np.array(mix_vals)

    elif spacing == 'lin':
        values = np.linspace(start=min_val, stop=1, num=n_mix_vals)

    elif spacing == 'log':

        if prefer_larger:
            values = 1 + min_val - np.logspace(start=0, stop=np.log10(min_val),
                                               num=n_mix_vals)
        else:
            values = np.logspace(start=np.log10(min_val), stop=0,
                                 num=n_mix_vals)

    return np.sort(values)


def get_pen_val_from_mix_seq(pen_val_max,
                             mix_vals,
                             pen_vals=None,
                             n_pen_vals=100,
                             pen_min_mult=1e-3,
                             pen_spacing='log',
                             second_val_max=None):
    """
    Sets up the pen_val tuning sequence for a mixed penalty that looks like

    pen_val * mix_val * primary_penalty(coef) + pen_val * (1 - mix_val) * second_penalty(coef)

    Parameters
    ----------
    pen_val_max: float
        The largest reasonable value for primary_penalty()'s penalty value.

    mix_vals: array-like or float
        The mixing parameter value or sequence of values.

    pen_vals: None, array-like
        (Optional) User specified tuning sequence for pen_val. If not provided, we create a default sequence.

    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence.

    pen_min_mult: float
        Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    second_val_max: None, float
        (Optional) Largest reasonable value of second_penalty()'s penalty value. This is only needed if 0 is included in the mix_val sequence.
    """

    # check if we are tuning of mix vals or a fixed mix_val was provided
    if isinstance(mix_vals, Number):
        tune_mix_vals = True
    else:
        tune_mix_vals = False
        mix_vals = np.array(mix_vals)

    if pen_vals is not None:  # user provided pen vals
        pen_vals = np.array(pen_vals)

    else:  # create penalty sequence

        # check we have all the info we need
        if min(mix_vals) < np.finfo(float).eps:
            if second_val_max is None:
                raise RuntimeError("If mix_val is zero then"
                                   "second_val_max must be provided ")

        if tune_mix_vals:
            # setup grid of pen vals for each mix_val

            n_mix_vals = len(mix_vals)
            pen_vals = np.zeros((n_mix_vals, n_pen_vals))
            for mix_idx in range(n_mix_vals):

                # largest pen val given this l1_ratio
                if mix_vals[mix_idx] < np.finfo(float).eps:
                    max_val = second_val_max
                else:
                    max_val = pen_val_max / mix_vals[mix_idx]

                # setup pen_val sequence
                pen_vals[mix_idx, :] = \
                    get_sequence_decr_max(max_val=max_val,
                                          min_val_mult=pen_min_mult,
                                          num=n_pen_vals,
                                          spacing=pen_spacing)

        else:
            # setup pen val sequence

            #
            max_val = pen_val_max / mix_vals
            pen_vals = get_sequence_decr_max(max_val=max_val,
                                             min_val_mult=pen_min_mult,
                                             num=n_pen_vals,
                                             spacing=pen_spacing)

    # ensure correct ordering
    if tune_mix_vals:
        assert pen_vals.ndim == 2
        assert pen_vals.shape[0] == len(mix_vals)

        # make sure pen vals are always in decreasing order
        for l1_idx in range(pen_vals.shape[0]):
            pen_vals[l1_idx, :] = np.sort(pen_vals[l1_idx, :])[::-1]

    else:
        # make sure pen vals are always in decreasing order
        pen_vals = np.sort(np.array(pen_vals))[::-1]
        pen_vals = pen_vals.reshape(-1)

    return pen_vals
