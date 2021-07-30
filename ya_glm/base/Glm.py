from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, _check_sample_weight, \
    FLOAT_DTYPES

from ya_glm.loss.LossConfig import get_loss_config
from ya_glm.autoassign import autoassign
from ya_glm.processing import process_X, deprocess_fit
from ya_glm.solver.default import get_default_solver


class Glm(BaseEstimator):
    """
    Base class for GLMs.

    Parameters
    ----------
    loss: str, ya_glm.LossConfig.LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See ya_glm.LossConfig for available loss functions.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Standardization means mean centering and scaling each column by its standard deviation. For the group lasso penalty an additional scaling is applied that scales each variable by 1 / sqrt(group size). Putting each variable on the same scale makes sense for fitting penalized models. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    solver: str, ya_glm.GlmSolver
        The solver used to solve the penalized GLM optimization problem. If this is set to 'default' we try to guess the best solver. Otherwise a custom solver can be provided by specifying a GlmSolver object.

    Attributes
    ----------
    coef_: array-like, shape (n_features, ) or (n_features, n_responses)
        The fitted coefficient vector or matrix (for multiple responses).

    intercept_: None, float or array-like, shape (n_features, )
        The fitted intercept.

    classes_: array-like, shape (n_classes, )
        A list of class labels known to the classifier.

    opt_data_: dict
        Data output by the optimization algorithm.
    """
    @autoassign
    def __init__(self, loss='lin_reg',
                 fit_intercept=True, standardize=False,
                 solver='default'):
        pass

    def _get_loss_config(self):
        """
        Returns the loss function config.

        Output
        ------
        loss: ya_glm.LossConfig.LossConfig
            The loss function config object.
        """
        return get_loss_config(loss=self.loss)

    @property
    def _estimator_type(self):
        """
        Type of the estimator.

        Output
        ------
        _estimator_type: str
            Either 'regressor' or 'classifier'.
        """
        return self._get_loss_config()._estimator_type

    def _get_solver(self):
        """
        Returns the solver config.

        Output
        ------
        solver: ya_glm.GlmSolver
            The solver config object.
        """

        if type(self.solver) == str and self.solver == 'default':
            # try to guess the best solver for our purposes
            # e.g. FISTA does not work for quantile regression loss
            return get_default_solver(loss=self._get_loss_config(),
                                      penalty=self._get_penalty_config())

        else:
            return self.solver

    def fit(self, X, y, sample_weight=None):
        """
        Fits the penalized GLM.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        self
            Fitted estimator.
        """

        # basic formattin check
        X, y, sample_weight = self._validate_data(X=X, y=y,
                                                  sample_weight=sample_weight)

        # run prefitting procedures including preprocessing the X, y data
        X_pro, y_pro, pre_pro_out, penalty_data =\
            self.prefit(X=X, y=y, sample_weight=sample_weight)

        # get the loss, penalty and solver config
        loss = self._get_loss_config()
        penalty = self._get_penalty_config()
        solver = self._get_solver()

        # possibly add information to the penalty
        # e.g. the initial coefficient for concave penalities
        if penalty_data is not None and len(penalty_data) > 0:
            penalty.set_data(penalty_data)

        # solve the optimzation problem!!!
        coef, intercept, out_data = \
            solver.solve(X=X_pro, y=y_pro,
                         loss=loss,
                         penalty=penalty,
                         fit_intercept=self.fit_intercept,
                         sample_weight=sample_weight)

        # set the fit coefficient e.g. undo preprocessing scaling
        self._set_fit(fit_out={'coef': coef,
                               'intercept': intercept,
                               'opt_data': out_data},
                      pre_pro_out=pre_pro_out)

        return self

    def prefit(self, X, y, sample_weight=None):
        """
        Preprocesses data and possibly performs other prefitting routines e.g. fitting an initial estimator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        X_pro: array-like, shape (n_samples, n_features)
            The processed covariate data.

        y_pro: array-like, shape (n_samples, )
            The processed response data.

        pro_pro_out: dict
            Data from preprocessing e.g. X_center, X_scale.

        penalty_data: None, dict
            Additional data that the penalty needs to know about.
        """
        # preproceess X, y
        X_pro, y_pro, pre_pro_out = \
            self.preprocess(X=X, y=y, sample_weight=sample_weight, copy=True)

        # by default we dont do any thing here
        penalty_data = None

        return X_pro, y_pro, pre_pro_out, penalty_data

    def _validate_data(self, X, y, sample_weight=None, accept_sparse=True):
        """
        Validates the X/y data. This should not change the raw input data, but may reformat the data (e.g. convert pandas to numpy).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.
        """

        X = check_array(X, accept_sparse=accept_sparse,
                        dtype=FLOAT_DTYPES)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # make sure y is numpy and of same dtype as X
        # TODO: do we actually want this for log_reg/multinomial?
        y = check_array(y, ensure_2d=False)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        # make sure X, y have same number of samples
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows!")

        return X, y, sample_weight

    def preprocess(self, X, y, sample_weight=None, copy=True, check_input=True):
        """
        Preprocesses the data for fitting. This method may transform the data e.g. centering and scaling X. If sample weights are provided then these are used for computing weighted means / standard deviations for standardization. For the group lasso penalty an additional scaling is applied that scales each variable by 1 / sqrt(group size).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

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
            Data from preprocessing e.g. X_center, X_scale.
        """
        groups = self.groups if hasattr(self, 'groups') else None

        X, out = process_X(X,
                           standardize=self.standardize,
                           groups=groups,
                           sample_weight=sample_weight,
                           copy=copy,
                           check_input=check_input,
                           accept_sparse=True,
                           allow_const_cols=not self.fit_intercept)

        # subclass should implement this
        y, y_out = self._process_y(X=X, y=y,
                                   sample_weight=sample_weight,
                                   copy=copy)
        out.update(y_out)

        return X, y, out

    def _set_fit(self, fit_out, pre_pro_out):
        """
        Sets the fit from the ouptut of the optimization algorithm.
        For example, this undoes any centering and scaling we have performed on the data so the fitted coefficient matches the raw input data.

        Parameters
        ----------
        fit_out: dict
            Contains the output of solve e.g.
            fit_out['coef'], fit_out['intercept'], fit_out['opt_data']

        pre_pro_out: None, dict
            Output of preprocess
        """
        coef = fit_out['coef']
        intercept = fit_out.pop('intercept', None)

        self.coef_, self.intercept_ = \
            deprocess_fit(coef=coef,
                          intercept=intercept,
                          pre_pro_out=pre_pro_out,
                          fit_intercept=self.fit_intercept)

        if 'opt_data' in fit_out:
            self.opt_data_ = fit_out['opt_data']

        # for classification models
        if 'classes' in pre_pro_out:
            self.classes_ = pre_pro_out['classes']

    def decision_function(self, X):
        """
        The GLM decision function i.e. z = X.T @ coef + interept

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The covariate data.

        Output
        ------
        z: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The decision function values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # TODO: for multi-response our coef_ is the transpose of sklearn's
        # convention. I think our choice of (n_features, n_responses)
        # Do we want to be stick with this choice?
        z = safe_sparse_dot(X, self.coef_,  # .T
                            dense_output=True)

        if hasattr(self, 'intercept_') and self.intercept_ is not None:
            z += self.intercept_

        return z

    def _more_tags(self):
        return {'requires_y': True}

    ################################
    # sub-classes should implement #
    ################################

    # this is set by the LossMixin
    def _process_y(self, X, y, sample_weight=None, copy=True):
        """
        Processing for the y data e.g. transform class labels to indicator variables for multinomial.

        Parameters
        ---------
        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample

        copy: bool
            Whether or not to copy the X/y arrays or modify them in place.

        Output
        ------
        y: array-like
            The possibly transformed response data.
        """
        # subclass should overwrite
        raise NotImplementedError

    def _get_penalty_config(self):
        """
        Gets the penalty config.

        Output
        ------
        penalty: ya_glm.PenaltyConfig.PenaltyConfig
            A penalty config object.
        """
        # subclass should implement!
        raise NotImplementedError

    def get_pen_val_max(self, X, y, sample_weight=None):
        """
        Returns the largest reasonable penalty parameter for a given dataset.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        pen_val_max: float
            Largest reasonable tuning parameter value.
        """
        # subclasses should implement!
        raise NotImplementedError
