from copy import deepcopy
import numpy as np

from ya_glm.utils import fit_if_unfitted
from ya_glm.utils import get_coef_and_intercept


class InitFitMixin:
    """
    Mixin for algorithms that need an initializer.

    Parameters
    ----------
    init: str, dict, estimator.
        If init='default', will use a default estimator.
        If init='zero', will initialize at zero.
        If init is a dict, will return self.init. If init is an estimator that is already fit, it will NOT be refit on the new data.
    """

    def _get_init_data(self, X, y=None, **fit_params):
        """
        Fits an initial estimator to the data. Will not fit if self.init is an already fit estimator.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        **fit_params:
            Keyword arguments to possibly be passed to a call to fit

        Output
        ------
        init_data: dict
            The data required to initialize the LLA algorithm
        """
        # user provided initial values
        if isinstance(self.init, dict):
            return deepcopy(self.init)

        elif self.init == 'zero':
            # initialize at zero
            coef_shape, intercept_shape = self._get_coef_intercept_shape(X, y)
            coef = np.zeros(coef_shape)

            if intercept_shape[0] == 0:
                intercept = 0
            else:
                intercept = np.zeros(intercept_shape)

            return {'coef': coef,
                    'intercept': intercept
                    }

        elif self.init == 'default':
            # initialize from defualt strategy
            defualt_init = self._get_default_init()

            # if isinstance(defualt_init, dict):
            #     # if defualt values are provided then return them
            #     return deepcopy(defualt_init)

            # else:
            # otherwise a defulat estimator was provided

            # possibly fit the defulat estimator
            init_est = fit_if_unfitted(defualt_init, X=X, y=y,
                                       **fit_params)

        else:

            # user provided an estimator
            init_est = fit_if_unfitted(self.init, X=X, y=y,
                                       **fit_params)

        coef, intercept = get_coef_and_intercept(init_est, copy=True,
                                                 error=True)


        return {'coef': coef,
                'intercept': intercept,
                'est': init_est}

    ################################
    # subclasses need to implement #
    ################################

    def _get_default_init(self):
        """
        Returns the default initial estimator.

        Output
        ------
        est:
            The default initializer.
        """
        raise NotImplementedError
