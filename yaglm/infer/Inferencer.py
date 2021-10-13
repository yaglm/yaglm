from numbers import Number
from pandas.core.dtypes.inference import is_array_like
from copy import deepcopy
import numpy as np

from yaglm.config.base import Config
from yaglm.autoassign import autoassign
from yaglm.config.penalty import Lasso
from yaglm.infer.dof import est_dof_support
from yaglm.utils import is_fitted
from yaglm.config.loss import get_loss_config


class Inferencer(Config):
    """
    An object that runs statistical inference routines for penalized GLMs. This includes estimating:

    - GLM exponential family scale parameter (if there is one)
    - degrees of freedom of estimated coefficient.

    Parameters
    ----------
    dof: str
        Method to estimator the number of degrees of freedom.

    scale: None, float, array-like, str
        Method to estimate the GLM scale parameter (e.g. linear regression noise standard deviation) if one is required. The scale parameter(s) can be manually set by providing a float or array-like

    Attributes
    ----------
    dof_: int, None
        The estimated degrees of freedom. If no dof estimation procedure is available this is None.


    scale_: None, float, array-like
        The estimated scale parameter. Only for GLMs that have scale parameters. If no scale estimation procedure is available this is None.

    scale_est_: ScaleEstimator
        The estimator object used to estimate the GLM scale parameter, if one was used.
    """

    @autoassign
    def __init__(self, dof='support', scale=None): pass

    def pre_fit(self, estimator, X, y, sample_weight=None):
        """
        Runs inferences procedures before fitting the estimator e.g. estimating the scale parameter.

        Parameters
        ----------
        estimator: Estimator
            The fit estimator object we want to run inference on.

        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: array-like, shape (n_samples, )
            The training sample weights.

        Output
        ------
        self
        """
        self.X_shape_ = X.shape

        ################################
        # estimate GLM scale parameter #
        ################################

        loss = get_loss_config(estimator.loss)
        if loss.has_scale:
            # TODO: we need to pass the loss to the scale estiamtor
            # e.g. even just to validate we have the right scale estimaor

            # if a float or array was provided use this value
            if isinstance(self.scale, Number):
                self.scale_ = deepcopy(self.scale)
            elif is_array_like(self.scale):
                self.scale_ = np.array(self.scale).reshape(-1)

            else:
                # TODO: do we want a copy here?
                scale_est = self.scale

                # fit the scale estimator if we have not already fit it
                if not is_fitted(scale_est):
                    scale_est.fit(X=X, y=y,
                                  sample_weight=sample_weight)

                self.scale_ = scale_est.scale_

    def after_fit(self, estimator, X, y, sample_weight=None):
        """
        Runs inferences procedures after fitting the estimator e.g. estimating the degrees of freedom.

        Parameters
        ----------
        estimator: Estimator
            The fit estimator object we want to run inference on.

        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, ) or (n_samples, n_responses)
            The training response data.

        sample_weight: array-like, shape (n_samples, )
            The training sample weights.

        Output
        ------
        self
        """
        # TODO-THINK-THROUGH: Do we want all inference procedures to use the raw data? Should they sometimes use the processed data?

        zero_tol = 1e-6

        #########################################
        # estimate number of degrees of freedom #
        #########################################
        if self.dof == 'support':

            if isinstance(estimator.fit_penalty_, Lasso):

                self.dof_ = est_dof_support(coef=estimator.coef_,
                                            intercept=estimator.intercept_,
                                            transform=None,
                                            zero_tol=zero_tol)

            else:
                # we don't currently support estimating the DoF for this model
                self.dof_ = None

        elif isinstance(self.dof, Number):
            # user provided DOF value
            self.dof_ = deepcopy(self.dof)

        else:
            raise NotImplementedError("Do don't currently support dof={}".
                                      format(self.dof))
        return self
