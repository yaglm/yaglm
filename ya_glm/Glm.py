from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, FLOAT_DTYPES

from scipy.sparse import diags
import numpy as np
from textwrap import dedent

from ya_glm.utils import is_multi_response
from ya_glm.autoassign import autoassign
from ya_glm.processing import process_X


class Glm(BaseEstimator):
    """
    Base class for Lasso generalized linear model.

    Parameters
    ----------
    fit_intercept: bool
        Whether or not to fit an intercept.

    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Here standardization means mean centering and scaling each column by its standard deviation. Putting each column on the same scale makes sense for fitting penalized models. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    opt_kws: dict
        Key word arguments to the optimization algorithm.
    """

    # subclasses should specify these
    # TODO: is this how we want to store this data?

    # specify the model type so we know what loss func to use
    _model_type = None

    # subclass should implement
    solve_glm = None

    @autoassign
    def __init__(self, fit_intercept=True, standardize=False, opt_kws={}):
        pass

    def fit(self, X, y):
        """
        Fits the GLM.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.
        """

        X, y = self._validate_data(X, y)

        # TODO: give user option to not copy, but need to think more
        # about this
        X, y, pre_pro_out = self._pre_process(X, y, copy=True)
        fit_data = self._compute_fit(X, y)
        self._set_fit(fit_data, pre_pro_out)
        return self

    def _validate_data(self, X, y, accept_sparse=False):
        """
        Validates the X/y data. This should not change the raw input data, but may reformat the data (e.g. convert pandas to numpy).


        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.
        """

        # TODO: waht to do about sparse???
        X = check_array(X, accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)

        # make sure y is numpy and of same dtype as X
        y = np.asarray(y, dtype=X.dtype)

        # make sure X, y have same number of samples
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows!")

        return X, y

    def _pre_process(self, X, y, copy=True):
        """
        Preprocesses the data for fitting. This method may transform the data e.g. centering and scaling X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        copy: bool
            Whether or not to copy the X/y arrays or modify them in place.

        Output
        ------
        X_pro, y_pro, pre_pro_out

        X_pro: array-like, shape (n_samples, n_features)
            The possibly transformed covariate data.

        y_pro: array-like, shape (n_samples, )
            The possibly transformed response data.

        pro_pro_out: dict
            The data used to preprocess the X/y data.
        """

        X, out = process_X(X,
                           standardize=self.standardize,
                           copy=copy,
                           check_input=False,
                           accept_sparse=False,  # TODO!
                           allow_const_cols=not self.fit_intercept)

        y, y_out = self._process_y(y, copy=copy)
        out.update(y_out)

        return X, y, out

    def _process_y(self, y, copy=True):
        """
        Parameters
        ---------
        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        Output
        ------
        y: array-like
            The possibly transformed response data.
        """
        raise NotImplementedError

    def _compute_fit(self, X, y):
        """
        Solve the GLM optimizaiton problem.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        Output
        ------
        fit_out: dict
            The output from fitting the model.

            fit_out['coef']: array-like, shape (n_features, )
                The fitted GLM coefficient.

            fit_out['intercept']: None or Float
                The fitted intercept.

            fit_out['opt_data']: dict
                Any output from the optimization algorithm e.g. number of iterations.
        """

        coef, intercept, opt_data = \
            self.solve_glm(X=X, y=y,
                           # loss_func=self._model_type,
                           # fit_intercept=self.fit_intercept,
                           # **self.opt_kws
                           **self._get_solve_glm_kws()
                           )

        fit_out = {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}
        return fit_out

    def _get_solve_glm_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """
        raise NotImplementedError

    def _set_fit(self, fit_out, pre_pro_out=None):
        """
        Sets the fit.

        Parameters
        ----------
        fit_out: dict
            Output of _compute_fit

        pre_pro_out: None, dict
            Output of _pre_process
        """

        # set coefficient
        coef = np.array(fit_out['coef'])
        is_mr = is_multi_response(coef)
        if not is_mr:
            coef = coef.ravel()

        # rescale coefficient
        if pre_pro_out is not None and 'X_scale' in pre_pro_out:
            # coef = coef / pre_pro_out['X_scale']
            coef = diags(1 / pre_pro_out['X_scale']) @ coef
        self.coef_ = coef

        # maybe set intercept
        if self.fit_intercept:
            intercept = fit_out['intercept']

            if pre_pro_out is not None and 'X_offset' in pre_pro_out:
                intercept -= coef.T @ pre_pro_out['X_offset']

            if pre_pro_out is not None and 'y_offset' in pre_pro_out:
                intercept += pre_pro_out['y_offset']

            self.intercept_ = intercept

        else:
            if is_mr:
                self.intercept_ = np.zeros(coef.shape[1])
            else:
                self.intercept_ = 0

        if 'opt_data' in fit_out:
            self.opt_data_ = fit_out['opt_data']

    def _decision_function(self, X):
        """
        The GLM decision function i.e. z = X.T @ coef + interept

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        Output
        ------
        z: array-like, shape (n_samples, )
            The decision function values.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        z = safe_sparse_dot(X, self.coef_, dense_output=True) \
            + self.intercept_

        # if hasattr(self, 'intercept_'):
        #     z += self.intercept_

        return z

    def decision_function(self, X):
        """
        The GLM decision function i.e. z = X.T @ coef + interept

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        Output
        ------
        z: array-like, shape (n_samples, )
            The decision function values.
        """
        return self._decision_function(X)

    def _more_tags(self):
        return {'requires_y': True}

    def get_pen_val_max(self, X, y):
        """
        Returns the largest reasonable penalty parameter from the processed training data. I.e. this is an lower bound such that any larger tuning parameter value will force the coefficient to be zero.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        Output
        ------
        pen_val_max: float
            Largest reasonable tuning parameter value.
        """
        # make sure we get set tuning parameters using the processed data
        # the optimization algorithm will actually see
        X_pro, y_pro, pre_pro_data = self._pre_process(X, y, copy=True)
        return self._get_pen_val_max_from_pro(X_pro, y_pro)

    def _get_pen_val_max_from_pro(self, X, y):
        raise NotImplementedError
