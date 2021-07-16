from copy import deepcopy
# from textwrap import dedent

from ya_glm.utils import fit_if_unfitted
from ya_glm.utils import get_coef_and_intercept
from ya_glm.processing import process_init_data


class InitMixin:
    """
    init

    _get_defualt_init

    _get_init_data_from_fit_est
    """

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

    def _get_defualt_init(self):
        raise NotImplementedError

    def _get_init_data_from_fit_est(self, est, X, y):
        raise NotImplementedError


class GlmWithInitMixin(InitMixin):

    def fit(self, X, y, sample_weight=None):

        # validate the data!
        X, y, sample_weight = self._validate_data(X, y,
                                                  sample_weight=sample_weight)

        # get data for initialization
        init_data = self.get_init_data(X, y)
        if 'est' in init_data:
            self.init_est_ = init_data['est']
            del init_data['est']

        # pre-process data
        X_pro, y_pro, pre_pro_out = self.preprocess(X, y,
                                                    sample_weight=sample_weight,
                                                    copy=True)

        # possibly process the init data e.g. shift/scale
        init_data_pro = process_init_data(init_data=init_data,
                                          pre_pro_out=pre_pro_out)

        # Fit!
        fit_out = self.compute_fit(X=X_pro, y=y_pro,
                                   init_data=init_data_pro,
                                   sample_weight=sample_weight)

        self._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
        return self

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

    def get_pen_val_max(self, X, y, init_data=None, sample_weight=None):
        if init_data is None:
            init_data = self.get_init_data(X, y, sample_weight=sample_weight)

        X_pro, y_pro, pre_pro_out = self.preprocess(X, y,
                                                    sample_weight=sample_weight,
                                                    copy=True)

        init_data_pro = process_init_data(init_data=init_data,
                                          pre_pro_out=pre_pro_out)

        return self._get_pen_val_max_from_pro(X=X_pro, y=y_pro,
                                              init_data=init_data_pro,
                                              sample_weight=sample_weight)

    def _get_pen_val_max_from_pro(self, X, y, init_data, sample_weight=None):
        raise NotImplementedError

    def compute_fit(self, X, y, init_data, sample_weight=None):
        raise NotImplementedError
