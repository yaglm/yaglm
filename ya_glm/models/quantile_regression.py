from sklearn.utils.validation import check_array

from ya_glm.models.linear_regression import LinRegMixin
from ya_glm.autoassign import autoassign


class QuantileRegMixin(LinRegMixin):

    is_multi_resp = False

    @autoassign
    def __init__(self, quantile=0.5): pass

    def get_loss_info(self):
        loss_type = 'quantile'
        loss_kws = {'quantile': self.quantile}

        return loss_type, loss_kws

    def _process_y(self, y, copy=True):
        return process_y_quantile(y,
                                  copy=copy,
                                  check_input=True)


def process_y_quantile(y, copy=True, check_input=True):
    """
    Process y for quantile regression.

    Parameters
    ----------
    y: array-like, shape (n_samples, )
        The response data.

    copy: bool
        Make sure y is copied and not modified in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, out

    y: array-like, shape (n_samples, )
        The responses
    """
    if check_input:
        y = check_array(y, copy=copy, ensure_2d=False)
    elif copy:
        y = y.copy(order='K')

    y = y.reshape(-1)

    return y, {}
