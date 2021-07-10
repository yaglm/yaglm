from sklearn.utils.validation import check_array
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.base import RegressorMixin
# from scipy.sparse.linalg import LinearOperator

import numpy as np

from ya_glm.cv.scoring import get_n_nonzero, Scorer


class LinRegMixin(RegressorMixin):
    """
    fit_intercept:

    normalize:

    copy_X:

    """
    def get_loss_info(self):
        loss_type = 'lin_reg'
        loss_kws = {}

        return loss_type, loss_kws

    def _process_y(self, y, copy=True):
        return process_y_lin_reg(y, standardize=self.standardize,
                                 copy=copy,
                                 check_input=True)

    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)


def process_y_lin_reg(y, standardize=False, copy=True, check_input=True):
    """
    Processes and possibly mean center the y data i.e. y - y.mean()

    Parameters
    ----------
    y: array-like, shape (n_samples, )
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

    y: array-like, shape (n_samples, )
        The possibly mean centered responses.

    out: dict
        The pre-processesing output data. If standardize=True this contains
        out['y_offset']: float
            The response mean.
    """
    if check_input:
        y = check_array(y, ensure_2d=False)
    elif copy:
        y = y.copy(order='K')

    y = y.reshape(-1)

    # mean center y
    out = {}
    if standardize:
        out['y_offset'] = y.mean()
        y -= out['y_offset']

    return y, out


# def center_scale_sparse(X, X_offset, X_scale):
#     X_offset_scale = X_offset / X_scale

#     def matvec(b):
#         return X.dot(b) - b.dot(X_offset_scale)

#     def rmatvec(b):
#         return X.T.dot(b) - X_offset_scale * np.sum(b)

#     X_centered = LinearOperator(shape=X.shape,
#                                 matvec=matvec,
#                                 rmatvec=rmatvec)

#     return X_centered


class LinRegScorer(Scorer):
    def __init__(self, verbosity=1, default='r2'):
        super().__init__(scorer=score_lin_reg,
                         verbosity=verbosity,
                         default=default)


def score_lin_reg(est, X, y, verbosity=1):
    """
    Scores a fitted linear regression model.

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features)
        The test X data.

    y_true: array-like, shape (n_samples, )
        The true responses.

    est:
        The fitted estimator.

    verbosity: int
        How much data to return.

    Output
    ------
    scores: dict
        Conatins the scores.
    """

    # get predictions
    y_pred = est.predict(X)

    out = {}
    out['r2'] = r2_score(y_true=y, y_pred=y_pred)

    if verbosity >= 1:
        out['n_nonzero'] = get_n_nonzero(est.coef_, zero_tol=1e-8)
        out['median_abs_err'] = median_absolute_error(y_true=y, y_pred=y_pred)

        out['l2_norm'] = np.linalg.norm(est.coef_.reshape(-1))

    return out
