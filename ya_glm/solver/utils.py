import numpy as np


def process_param_path(lasso_pen_seq=None, ridge_pen_seq=None, check_decr=True):

    if check_decr:
        if lasso_pen_seq is not None:
            assert all(np.diff(lasso_pen_seq) <= 0)

        if ridge_pen_seq is not None:
            assert all(np.diff(ridge_pen_seq) <= 0)

    if lasso_pen_seq is not None and ridge_pen_seq is not None:
        assert len(lasso_pen_seq) == len(ridge_pen_seq)

        param_path = [{'lasso_pen_val': lasso_pen_seq[i],
                       'ridge_pen_val': ridge_pen_seq[i]}
                      for i in range(len(lasso_pen_seq))]

    elif lasso_pen_seq is not None:
        param_path = [{'lasso_pen_val': lasso_pen_seq[i]}
                      for i in range(len(lasso_pen_seq))]

    elif ridge_pen_seq is not None:
        param_path = [{'ridge_pen_val': ridge_pen_seq[i]}
                      for i in range(len(ridge_pen_seq))]

    else:
        raise ValueError("One of lasso_pen_seq, ridge_pen_seq should be provided ")

    return param_path
