from scipy.sparse import issparse
import numpy as np

from ya_glm.extmath import weighted_mean_std
from ya_glm.sparse_utils import center_scale_sparse, is_sparse_or_lin_op


def center_Xy(X, y, sample_weight=None, copy=True):
    """
    Returns centered versions of the X, y data.

    Parameters
    ---------
    X: array-like, shape (n_samples, n_features)
        The covariate data.

    y: array-like, shape (n_samples, )
        The response data.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    copy: bool
        Copy data matrix or center in place.

    Output
    ------
    X_pro, y_pro, out_data

    X_pro: array-like, shape (n_samples, n_features)
        The centered covariate data.

    y_pro: array-like, shape (n_samples, )
        The centered response data.

    out: dict
        Contains the X_offset and y_offset terms.
    """
    
    if copy:
        if issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

        y = y.copy()

    # TODO: make mean only version of this
    X_offset, X_scale = weighted_mean_std(X,
                                          sample_weight=sample_weight,
                                          ddof=0)
    
    y_offset = np.average(a=y, axis=0, weights=sample_weight)
    
    # TODO: check/make this works for linear operators
    if is_sparse_or_lin_op(X):
        X = center_scale_sparse(X, X_offset=X_offset)
    else:
        X -= X_offset
    
    y -= y_offset
    
    return X, y, {'X_offset': X_offset, 'y_offset': y_offset}
    
    
def fake_intercept_via_centering(coef, X_offset, y_offset):
    """
    Sets the fake intercept from the estimated coefficient and initial centering data in the same way sklearn.linear_model.Lasso does.

    For unpenalized linear regression we can center the X/y data, compute the coefficient then obtain the intercept using the formula implemented in this function. We can do the same operators for peanlized linear regression (e.g. where we solve for the penalized coefficeint, but leave the intercept out of the model). If the X, y data are uncentered this will not give the exactly correct intercept in the penalized case, but it's a reasonable approximation.
    """
    return y_offset - coef.T @ X_offset
