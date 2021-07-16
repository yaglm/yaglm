from sklearn.utils.validation import check_array
import numpy as np

from ya_glm.glm_loss.linear_regression import LinRegMixin


class LinRegMultiRespMixin(LinRegMixin):

    is_multi_resp = True

    def get_loss_info(self):
        loss_type = 'lin_reg_mr'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, sample_weight=None, copy=True):
        return process_y_lin_reg_mr(y,
                                    standardize=self.standardize,
                                    sample_weight=sample_weight,
                                    copy=copy,
                                    check_input=True)


def process_y_lin_reg_mr(y, standardize=False, sample_weight=None,
                         copy=True, check_input=True):
    """
    Processes and possibly mean center the y data i.e. y - y.mean()

    Parameters
    ----------
    y: array-like, shape (n_samples, n_responses)
        The response data.

    sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

    standardize: bool
        Whether or not to mean center.

    copy: bool
        Copy data matrix or standardize in place.

    check_input: bool
        Whether or not we should validate the input.

    Output
    ------
    y, out

    y: array-like, shape (n_samples, n_responses)
        The possibly mean centered responses.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['y_offset']: array-like, shape (n_responses, )
            The response mean.
    """

    if check_input:
        y = check_array(y, copy=copy, ensure_2d=True)
    elif copy:
        y = y.copy(order='K')

    # make sure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # mean center y
    out = {}
    if standardize:
        out['y_offset'] = np.average(a=y, axis=0, weights=sample_weight)
        y -= out['y_offset']

    return y, out
