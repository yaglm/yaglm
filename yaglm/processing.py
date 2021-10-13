import numpy as np
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from scipy.sparse import issparse, diags
from copy import deepcopy

from yaglm.utils import is_multi_response
from yaglm.extmath import weighted_mean_std
from yaglm.sparse_utils import center_scale_sparse, is_sparse_or_lin_op, \
    safe_norm


def process_X(X, fit_intercept=True,
              standardize=False, groups=None, sample_weight=None, copy=True,
              check_input=True, accept_sparse=True):
    """
    Processes and possibly standardize the X feature matrix. If standardize=True then the coulmns are scaled such that their euclidean norm is equal to sqrt(n_samples). If additionally fit_intercept=True, the columns of X are first mean centered before scaling.

    If grouops is provided an additional scaling is applied that scales each variable by 1 / sqrt(group size).

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    fit_intercept: bool
        Whether or not we fit an intercept. If fit_intercept=False, standardize=True then we replace the standard deviation with the L2 norms.

    standardize: bool
        Whether or not to mean center then scale the columns of X.

    groups: None, list of lists
        (Optional) list of feature groups. If groups is provided and we apply standardization this applies and additional 1 / sqrt(group_size) scaling to feature.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    copy: bool
        Copy data matrix or standardize in place.

    check_input: bool
        Whether or not we should validate the input.

    accept_sparse : str, bool or list/tuple of str
        See sklearn.utils.validation.check_array.

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

    if standardize:

        if fit_intercept:

            # compute mean and standard deviations of each feature
            # TODO: this computes a weighted STD so the columns will be be perfectly norm 1.
            X_offset, X_scale = weighted_mean_std(X,
                                                  sample_weight=sample_weight,
                                                  ddof=0)

        else:
            X_offset = None

            # L2 norms of columns
            X_scale = safe_norm(X, axis=0)

            # make the norm of the columns be sqrt(n_samples)
            X_scale /= np.sqrt(X.shape[0])

        # columns with zero scale
        zero_scale_mask = X_scale <= np.finfo(float).eps

        # adjust scaling for groups
        if groups is not None:

            # compute sqrts of group sizes
            group_size_sqrts = np.array([np.sqrt(len(grp_idxs))
                                         for grp_idxs in groups])

            # sacle each feature by inverse sqrt of group size
            for g, grp_idxs in enumerate(groups):
                for feat_idx in grp_idxs:
                    X_scale[feat_idx] /= group_size_sqrts[g]

        # for columns with zero scale reset scale to 1
        X_scale[zero_scale_mask] = 1

        # data to return
        out['X_scale'] = X_scale
        if X_offset is not None:
            out['X_offset'] = X_offset

        # modify X
        if is_sparse_or_lin_op(X):
            X = center_scale_sparse(X, X_offset=X_offset, X_scale=X_scale)

        else:
            # mean center then scale
            if X_offset is not None:
                X -= X_offset

            X = X @ diags(1.0 / X_scale)

    return X, out


def process_groups(groups, n_features):
    return [np.array(grp_idxs).astype(int) for grp_idxs in groups]


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


def check_estimator_type(estimator, valid_class):
    """
    Checks if an estimator is an instance of a given subclass.
    Throws an error if it is not.
    """

    if not isinstance(estimator, valid_class):
        raise ValueError("estimator is wrong type. " \
                         "Got {} but should be {}".
                         format(type(estimator), valid_class))
