from copy import deepcopy
# from textwrap import dedent

from ya_glm.utils import fit_if_unfitted
from ya_glm.utils import get_coef_and_intercept


class GlmWithInitMixin:

    def get_init_data(self, X, y=None, **fit_params):
        """
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

        elif self.init == 'default':
            # initialize from defualt strategy
            defualt_init = self._get_defualt_init()

            if isinstance(defualt_init, dict):
                # if defualt values are provided then return them
                return deepcopy(defualt_init)

            else:
                # otherwise a defulat estimator was provided

                # possibly fit the defulat estimator
                default_init = fit_if_unfitted(defualt_init, X=X, y=y,
                                               **fit_params)
                return self._get_init_data_from_fit_est(est=default_init)

        else:

            # user provided an estimator
            init_est = fit_if_unfitted(self.init, X=X, y=y,
                                       **fit_params)
            return self._get_init_data_from_fit_est(est=init_est)

    def _get_init_data_from_fit_est(self, est):
        out = {}
        coef, intercept = get_coef_and_intercept(est, copy=True, error=True)

        out['coef'] = coef
        if self.fit_intercept:
            out['intercept'] = intercept
        else:
            out['intercept'] = None

        out['est'] = est

        return out

    def _get_defualt_init(self):
        raise NotImplementedError
