from sklearn.utils.validation import check_array, column_or_1d
from textwrap import dedent

from ya_glm.glm_loss.linear_regression import LinRegMixin
from ya_glm.autoassign import autoassign

_quant_reg_params = dedent("""
    quantile: float
        The loss function quantile.
    """)


class QuantileRegMixin(LinRegMixin):

    is_multi_resp = False

    _loss_descr = dedent("""
    Quantile regression with loss function L(z, y) = rho_{q}(z - y) where the tilted L1 loss is given by

    rho_q(r) = q if r <= 0
    rho_q(r) = 1 - q if r > 0
    """)

    _params_descr = _quant_reg_params

    @autoassign
    def __init__(self, quantile=0.5): pass

    def get_loss_info(self):
        loss_type = 'quantile'
        loss_kws = {'quantile': self.quantile}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_quantile(y,
                                  sample_weight=sample_weight,
                                  copy=copy,
                                  check_input=True)


class QuantileRegMultiRespMixin(LinRegMixin):

    is_multi_resp = True

    _loss_descr = dedent("""
    Multiple response quantile regression with loss function L(z, y) = sum_{j=1}^{n_responses} rho_{q}(z_j - y_j) where the tilted L1 loss is given by

    rho_q(r) = q if r <= 0
    rho_q(r) = 1 - q if r > 0
    """)

    _params_descr = _quant_reg_params

    @autoassign
    def __init__(self, quantile=0.5): pass

    def get_loss_info(self):
        loss_type = 'quantile_mr'
        loss_kws = {'quantile': self.quantile}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_quantile(y,
                                  sample_weight=sample_weight,
                                  copy=copy,
                                  check_input=True)


def process_y_quantile(y, sample_weight=None, copy=True, check_input=True):
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
        y = column_or_1d(y, warn=True)
    elif copy:
        y = y.copy(order='K')

    y = y.reshape(-1)

    return y, {}


def process_y_quantile_mr(y, sample_weight=None, copy=True, check_input=True):
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
        y = check_array(y, copy=copy, ensure_2d=True)
    elif copy:
        y = y.copy(order='K')

    y = y.reshape(-1)

    return y, {}
