import numpy as np
from ya_glm.base.Glm import PenGlm
from ya_glm.base.GlmWithInit import GlmWithInitMixin

from ya_glm.pen_max.lasso import get_pen_max

from ya_glm.init_signature import keep_agreeable
from ya_glm.opt.penalty.concave_penalty import get_penalty_func
from ya_glm.processing import process_init_data


# TODO: the way we set the adpative weights is a little ugly
# let's see if we can figure out a better solution.
# this is difficult because we need to know the original data (to get n_samples)
# but this needs to work with cross-validation where we clone the estimator
# this destroy any attributed derived from the original data


class GlmAdaptiveLassoBase(GlmWithInitMixin, PenGlm):

    def fit(self, X, y, sample_weight=None):

        # get the adaptive weights and preprocessed data
        adpt_weights, X_pro, y_pro, pre_pro_out, init_data_pro = \
            self._get_adpt_weights_and_pro_data(X, y, sample_weight)

        ##########################
        # solve the GLM problem! #
        ##########################

        kws = self._get_solve_kws()
        if sample_weight is not None:
            kws['sample_weight'] = sample_weight
        kws['lasso_weights'] = adpt_weights

        coef, intercept, opt_data = self.solve_glm(X=X, y=y, **kws)

        fit_out = {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}
        self._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
        self.adpt_weights_ = adpt_weights

        return self

    def _get_adpt_weights_and_pro_data(self, X, y, sample_weight=None):
        # validate the data!
        X, y, sample_weight = self._validate_data(X, y,
                                                  sample_weight=sample_weight)

        if self.adpt_weights is None:

            # get data for initialization if we have not already provided
            # the adaptive weights
            init_data = self.get_init_data(X, y)
            if 'est' in init_data:
                self.init_est_ = init_data['est']
                del init_data['est']

        else:
            init_data = None
            adpt_weights = self.adpt_weights

        # pre-process data for fitting
        X_pro, y_pro, pre_pro_out = self.preprocess(X, y,
                                                    sample_weight=sample_weight,
                                                    copy=True)

        if self.adpt_weights is None:
            # if we have not already provided the adpative weights
            # then compute them now

            # possibly process the init data e.g. shift/scale
            init_data_pro = process_init_data(init_data=init_data,
                                              pre_pro_out=pre_pro_out)

            adpt_weights = \
                self._get_adpt_weights_from_pro_init(init_data=init_data_pro,
                                                     n_samples=X.shape[0])

        else:
            init_data_pro = None

        return adpt_weights, X_pro, y_pro, pre_pro_out, init_data_pro

    def _get_adpt_weights_from_pro_init(self, init_data, n_samples=None):
        """
        Gets the adaptive lasso weights from the processed init data
        """
        coef = np.array(init_data['coef'])
        transform = self._get_coef_transform()
        t = transform(coef)

        if type(self.pertub_init) == str and self.pertub_init == 'n_samples':
            t += 1 / n_samples

        elif self.pertub_init is not None:
            t += self.pertub_init

        # Setup penalty function
        penalty_func = get_penalty_func(pen_func=self.pen_func,
                                        pen_val=1,
                                        pen_func_kws=self.pen_func_kws)
        weights = penalty_func.grad(t)
        return weights

    def _get_pen_max_lasso(self, X, y, init_data, sample_weight=None):

        # get the adaptive weights and processed data
        adpt_weights, X_pro, y_pro, pre_pro_out, init_data_pro = \
            self._get_adpt_weights_and_pro_data(X, y, sample_weight)

        loss_func, loss_kws = self.get_loss_info()
        pen_kind = self._get_penalty_kind()

        kws = {'X': X,
               'y': y,
               'fit_intercept': self.fit_intercept,
               'loss_func': loss_func,
               'loss_kws': loss_kws,
               'weights': adpt_weights,
               'sample_weight': sample_weight
               }

        if pen_kind == 'group':
            kws['groups'] = self.groups

        return get_pen_max(pen_kind, **kws)

    def _kws_for_default_init(self, c=None):
        """
        Returns the keyword arguments for the default initialization estimator.

        Parameters
        ----------
        c: None, class
            If a class is provided we only return keyword arguemnts that
            aggree with c.__init__
        """

        keys = ['fit_intercept', 'standardize', 'opt_kws',
                'ridge_weights', 'tikhonov',
                'groups', 'multi_task', 'nuc']
        return {k: self.__dict__[k] for k in keys}


class AdptCVMixin:
    def _pre_fit(self, X, y, init_data, estimator, sample_weight=None):
        """
        Sets the adaptive weights parameter.
        """

        # get the adaptive weights and preprocessed data
        adpt_weights, X_pro, y_pro, pre_pro_out, init_data_pro = \
            estimator._get_adpt_weights_and_pro_data(X, y, sample_weight)

        estimator.set_params(adpt_weights=adpt_weights)
        self.adpt_weights_ = adpt_weights
        return estimator
