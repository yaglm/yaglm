from sklearn.utils.validation import check_array

from ya_glm.models.linear_regression import LinRegMixin


class LinRegMultiResponseMixin(LinRegMixin):
    _model_type = 'lin_reg_mr'

    def _process_y(self, y, copy=True):
        return process_y_lin_reg_mr(y, standardize=self.standardize,
                                    copy=copy,
                                    check_input=True)


def process_y_lin_reg_mr(y, standardize=False, copy=True, check_input=True):
    """
    Processes and possibly mean center the y data i.e. y - y.mean()

    Parameters
    ----------
    y: array-like, shape (n_samples, n_responses)
        The response data.

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
        y = check_array(y, ensure_2d=True)
    elif copy:
        y = y.copy(order='K')

    # make sure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # mean center y
    out = {}
    if standardize:
        out['y_offset'] = y.mean(axis=0)
        y -= out['y_offset']

    return y, out
