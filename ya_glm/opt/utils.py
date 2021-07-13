from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
import numpy as np


def decat_coef_inter_vec(cat):
    """
    De-concatenates a vector of the coefficient and intercept.

    Parameters
    ----------
    cat: array-like, shape (n_features + 1, )
        The concatenation of a coefficeint vector and intercept where
        the intercept is the first coordinate.

    Output
    -----
    coef, intercept

    coef: array-like, shape (n_features, )
        The coefficient vector.

    intercept: float
        The intercept.
    """
    return cat[1:], cat[0]


def decat_coef_inter_mat(cat):
    """
    De-concatenates a matrix of the coefficient and intercept.

    Parameters
    ----------
    cat: array-like, shape (n_features + 1, n_responses)
        The concatenation of a coefficeint matrix and intercept vector where
        the intercept is the first row.

    Output
    -----
    coef, intercept

    coef: array-like, shape (n_features, n_responses)
        The coefficient vector.

    intercept: array-like, shape (n_responses, )
        The intercept.
    """
    return cat[1:, :], cat[0, :]


def safe_data_mat_coef_dot(X, coef, fit_intercept=True):
    """
    Computes X @ coef[1:] + coef[0] or X @ coef depending on if there is an intercept.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Data matrix

    coef: array-like, shape (n_features,) or  (n_features + 1,)
        Coefficients. If there is an intercept it sholuld be in the first coordinate.

    fit_intercept: bool
        Whether or not the first coordinate of coef is the intercept.

    Output
    ------
    z: array-like, (n_samples, )
    """
    if fit_intercept:
        coef, intercept = decat_coef_inter_vec(coef)
        return safe_sparse_dot(X, coef, dense_output=True) + intercept
    else:
        return safe_sparse_dot(X, coef, dense_output=True)


def safe_data_mat_coef_mat_dot(X, coef, fit_intercept=True):
    """
    Computes X @ coef[1:, :] + coef[0, :] or X @ coef depending on if there is an intercept.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Data matrix

    coef: array-like, shape (n_features, n_responses) or  (n_features + 1, n_responses)

        Coefficients. If there is an intercept it sholuld be in the first coordinate.

    fit_intercept: bool
        Whether or not the first coordinate of coef is the intercept.

    Output
    ------
    z: array-like, (n_samples, )
    """
    if fit_intercept:
        coef, intercept = decat_coef_inter_mat(coef)
        return safe_sparse_dot(X, coef, dense_output=True) + intercept

    else:
        return safe_sparse_dot(X, coef, dense_output=True)


# def safe_vec_coef_dot(vec, coef, fit_intercept=True):
#     if fit_intercept:
#         shift = 1
#     else:
#         shift = 0

#     return sum(vec[i] * coef[shift + i] for i in range(len(vec)))

def sign_never_0(x):
    """
    The sign function where the output is either +1 or -1 i.e. will not return a 0.
    """
    s = np.sign(x)

    if hasattr(s, '__len__'):
        s[s == 0] = 1

    else:
        if s == 0:
            s = 1

    return s


def safe_vectorize(pyfunc, *args, **kwargs):
    """
    Same as np.vectorize, but ensures the otype is a float. This prevents very bizare behavior where np.vectorize thinkgs something is an int when it should be a float.

    https://stackoverflow.com/questions/26316357/numpy-vectorize-returns-incorrect-values

    See np.vectorize for documentation
    """
    return np.vectorize(pyfunc=pyfunc, otypes=[np.float], *args, **kwargs)


def safe_entrywise_mult(A, B):
    """
    Safe entrywise multiplication of two arrays where one or both may be sparse.
    """
    if issparse(A):
        return A.multiply(B)
    elif issparse(B):
        return B.multiply(A)
    else:
        return A * B
