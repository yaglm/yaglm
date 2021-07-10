import numpy as np
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from scipy.sparse import issparse, diags
from copy import deepcopy

from ya_glm.utils import is_multi_response


def process_X(X, standardize=False, copy=True,
              check_input=True, accept_sparse=False,
              allow_const_cols=True):
    """
    Processes and possibly standardize the X feature matrix.
    Here standardization means mean centering then scaling by
    the standard deviation for each column.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    standardize: bool
        Whether or not to mean center then scale by the standard deviations.

    copy: bool
        Copy data matrix or standardize in place.

    check_input: bool
        Whether or not we should validate the input.

    accept_sparse : str, bool or list/tuple of str
        See sklearn.utils.validation.check_array.

    allow_const_cols: bool
        Whether or not to allow constant columns.

    Output
    ------
    X, out

    X: array-like, shape (n_samples, n_features)
        The possibly standardized covariate data.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['X_offset']: array-like, shape (n_features, )
            The column means.

        out['X_scale']: array-like, shape (n_features, )
            The column stds. Note if std=0 we set X_scale to 1.
    """

    out = {}

    # input validation and copying
    if check_input:
        X = check_array(X, copy=copy,
                        accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)
    elif copy:
        if issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

    # compute column STDs
    if standardize or not allow_const_cols:
        X_scale = X.std(axis=0)
        const_cols = X_scale <= np.finfo(float).eps

        # for constant columns, put their scale as 1
        X_scale[const_cols] = 1

    # check for constant columns
    if not allow_const_cols:
        if sum(const_cols) > 0:
            raise ValueError("Constant column detected")

    # center by feature means and scale by feature stds
    if standardize:
        X_offset = X.mean(axis=0)

        out['X_scale'] = X_scale
        out['X_offset'] = X_offset

        if issparse(X):
            raise NotImplementedError
            # TODO: add this

        else:
            # mean center then scale
            X -= X_offset
            X = X @ diags(1.0 / X_scale)

    return X, out


def deprocess_fit(coef, intercept, pre_pro_out, fit_intercept):
    """
    The X may be centered and scaled and the y data may be centered before fitting. This function undoes the processing on the fit coefficient/intercept so the new coef/intercept match the scale of the original data.

    Parameters
    ----------
    coef: array-like
        The coefficient fit on the processed data.

    intercept: None, float, array-like
        The intercept fit on the processed data.

    pre_pro_out: dict
        The output of preprocessing.

    fit_intercept: bool
        Whether or not we fit an intercept.

    Output
    ------
    coef, intercept

    """
    # set coefficient
    coef = np.array(coef)
    is_mr = is_multi_response(coef)
    if not is_mr:
        coef = coef.ravel()

    # rescale coefficient
    if pre_pro_out is not None and 'X_scale' in pre_pro_out:
        # coef = coef / pre_pro_out['X_scale']
        coef = diags(1 / pre_pro_out['X_scale']) @ coef

    # maybe set intercept
    if fit_intercept:

        if pre_pro_out is not None and 'X_offset' in pre_pro_out:
            intercept -= coef.T @ pre_pro_out['X_offset']

        if pre_pro_out is not None and 'y_offset' in pre_pro_out:
            intercept += pre_pro_out['y_offset']

    else:
        if is_mr:
            intercept = np.zeros(coef.shape[1])
        else:
            intercept = 0

    return coef, intercept


def process_init_data(init_data, pre_pro_out):
    """
    Process the initialization data. E.g. if we center/scale our X data before fitting we should transform the coef/intercept initializers to match the center/scaled data.

    This should not modify init_data

    Parameters
    ----------
    init_data: dict

    pre_pro_out: dict

    Output
    ------
    init_data_pro: dict
    """
    # TODO: double check this

    init_data_pro = {}

    # coefficient
    coef = np.array(init_data['coef'])
    is_mr = is_multi_response(coef)
    if not is_mr:
        coef = coef.ravel()

    # rescale coefficient
    if pre_pro_out is not None and 'X_scale' in pre_pro_out:
        coef = diags(pre_pro_out['X_scale']) @ coef

    init_data_pro['coef'] = coef

    # intercept
    if 'intercept' in init_data:
        intercept = deepcopy(init_data['intercept'])

        if intercept is not None:
            if pre_pro_out is not None and 'X_offset' in pre_pro_out:
                intercept += coef.T @ pre_pro_out['X_offset']

            if pre_pro_out is not None and 'y_offset' in pre_pro_out:
                intercept -= pre_pro_out['y_offset']

        init_data_pro['intercept'] = intercept

    return init_data_pro


def check_Xy(X, y):
    """
    Make sure X and y have the same number of samples and same data type.
    """
    y = np.asarray(y, dtype=X.dtype)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of rows!")

    return X, y

# TODO: where does this go?
def process_weights_group_lasso(groups, weights=None):
    """
    Processes the weights arguments for group lasso.

    Parameters
    ----------
    groups: list of array-like

    weights: str or array-like
        If weights == 'size' then we return 1/sqrt(group sizes)
    Output
    ------
    weights: None or array-like of floats
    """
    if weights is None:
        return None

    if weights == 'size':
        group_sizes = np.array([len(grp) for grp in groups])
        weights = 1 / np.sqrt(group_sizes)
    else:
        weights = weights


def check_estimator_type(estimator, valid_class):
    """
    Checks if an estimator is an instance of a given subclass.
    Throws an error if it is not.
    """

    if not isinstance(estimator, valid_class):
        raise ValueError("estimator is wrong type. " \
                         "Got {} but should be {}".
                         format(type(estimator), valid_class))
