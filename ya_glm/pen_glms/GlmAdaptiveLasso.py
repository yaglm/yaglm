from textwrap import dedent

from ya_glm.base.Glm import Glm
from ya_glm.base.GlmAdptLasso import GlmAdaptiveLassoBase, AdptCVMixin
from ya_glm.base.GlmCVWithInit import GlmCVWithInitSinglePen, GlmCVWithInitENet
from ya_glm.cv.CVPath import CVPathMixin
from ya_glm.cv.ENetCVPath import ENetCVPathMixin
from ya_glm.cv.CVGridSearch import CVGridSearchMixin


from ya_glm.init_signature import add_from_classes
from ya_glm.utils import maybe_add, lasso_and_ridge_from_enet
from ya_glm.processing import check_estimator_type
from ya_glm.make_docs import merge_param_docs


_adpt_lasso_params = dedent("""
pen_val: float
    The penalty value.

pen_func: str
    The concave function used to determine the adaptive weights.

pen_func_kws: dict
    Keyword arguments to the function used to determine the adpative weights.

init: str, estimator, dict with key ['coef'].
    The initial coefficient estimate that is used to determine the adaptive weights. If a fitted estimator or dict is provided the coefficient will be exctracted. If an unfitted estimator is provided the coefficient will be fitted from the data. If 'default' then a default initializer will be used.

pertub_init: None, float, str
    Possibly add a small value to the absolute values of the initial coefficient. If pertub_init='n_samples' then the purturbation value is 1 / n_samples.

groups: None, list of ints
    Optional groups of variables. If groups is provided then each element in the list should be a list of feature indices. Variables not in a group are not penalized.

ridge_pen_val: None, float
    Penalty strength for an optional ridge penalty.

ridge_weights: None, array-like shape (n_featuers, )
    Optional features weights for the ridge peanlty.

tikhonov: None, array-like (K, n_features)
    Optional tikhonov matrix for the ridge penalty. Both tikhonov and ridge weights cannot be provided at the same time.

adpt_weights: None, array-like
    Optional user specified adpative weights that are used instead of determining them from 'init'. These are the exact weights used and will not be be processed (e.g. scaled if standardization is used).

    """)


class GlmAdaptiveLasso(GlmAdaptiveLassoBase):

    solve_glm = None

    _pen_descr = dedent("""
    Adaptive Lasso or group lasso penalty with an optional ridge penalty.
    """)

    _params_descr = merge_param_docs(_adpt_lasso_params, Glm._params_descr)

    @add_from_classes(Glm)
    def __init__(self,
                 pen_val=1,
                 pen_func='log',  # TODO: is this the name we want?
                 pen_func_kws={},
                 init='default',
                 pertub_init='n_samples',
                 groups=None,
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None,
                 adpt_weights=None,  # TODO: we do this for tuning, perhaps better solution?
                 ): pass

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """

        if self.ridge_weights is not None and self.tikhonov is not None:
            raise ValueError("Both ridge weigths and tikhonov"
                             "cannot both be provided")

        loss_func, loss_kws = self.get_loss_info()

        kws = {'loss_func': loss_func,
               'loss_kws': loss_kws,

               'fit_intercept': self.fit_intercept,
               **self.opt_kws,

               'lasso_pen': self.pen_val,
               }

        ###################################
        # potential extra Lasso arguments #
        ###################################

        # let's only add these if they are not None
        # this way we can use solvers that doesn't have these kws
        extra_kws = {
                     'ridge_pen': self.ridge_pen_val,
                     'ridge_weights': self.ridge_weights,
                     'tikhonov': self.tikhonov
                     }

        kws = maybe_add(kws, **extra_kws)

        ##################################
        # potential lasso type arguments #
        ##################################

        pen_kind = self._get_penalty_kind()
        if pen_kind == 'group':
            kws['groups'] = self.groups

        elif pen_kind == 'multi_task':
            kws['L1to2'] = True

        elif pen_kind == 'nuc':
            kws['nuc'] = True

        return kws

    def get_pen_val_max(self, X, y, init_data, sample_weight=None):
        return self.\
            _get_pen_max_lasso(X, y, init_data, sample_weight=sample_weight)


class GlmAdaptiveLassoCVPath(AdptCVMixin, CVPathMixin, GlmCVWithInitSinglePen):

    _cv_descr = dedent("""
    Tunes the Adaptive Lasso penalty parameter via cross-validation using a path algorithm.
    """)

    def _get_solve_path_kws(self):
        if not hasattr(self, 'pen_val_seq_'):
            raise RuntimeError("pen_val_seq_ has not yet been set")

        kws = self.estimator._get_solve_kws()
        del kws['lasso_pen']
        kws['lasso_pen_seq'] = self.pen_val_seq_

        if hasattr(self, 'adpt_weights_'):
            kws['lasso_weights'] = self.adpt_weights_
        else:
            raise ValueError("adpt_weights has not been set")

        return kws

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmAdaptiveLasso)


class GlmAdaptiveLassoCVGridSearch(AdptCVMixin, CVGridSearchMixin,
                                   GlmCVWithInitSinglePen):

    _cv_descr = dedent("""
        Tunes the Adaptive Lasso penalty parameter via cross-validation.
        """)

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmAdaptiveLasso)


_adpt_enet_params = dedent("""
pen_val: float
    The penalty strength (corresponds to lambda in glmnet)

l1_ratio: float
    The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

pen_func: str
    The concave function used to determine the adaptive weights.

pen_func_kws: dict
    Keyword arguments to the function used to determine the adpative weights.

init: str, estimator, dict with key ['coef'].
    The initial coefficient estimate that is used to determine the adaptive weights. If a fitted estimator or dict is provided the coefficient will be exctracted. If an unfitted estimator is provided the coefficient will be fitted from the data. If 'default' then a default initializer will be used.

pertub_init: None, float, str
    Possibly add a small value to the absolute values of the initial coefficient. If pertub_init='n_samples' then the purturbation value is 1 / n_samples.

groups: None, list of ints
    Optional groups of variables. If groups is provided then each element in the list should be a list of feature indices. Variables not in a group are not penalized.

ridge_weights: None, array-like shape (n_featuers, )
    Optional features weights for the ridge peanlty.

tikhonov: None, array-like (K, n_features)
    Optional tikhonov matrix for the ridge penalty. Both tikhonov and ridge weights cannot be provided at the same time.

adpt_weights: None, array-like
    Optional user specified adpative weights that are used instead of determining them from 'init'. These are the exact weights used and will not be be processed (e.g. scaled if standardization is used).

    """)


class GlmAdaptiveENet(GlmAdaptiveLassoBase):
    solve_glm = None

    _pen_descr = dedent("""
    Adaptive ElasticNet or group ElasticNet.
    """)

    _params_descr = merge_param_docs(_adpt_enet_params, Glm._params_descr)

    @add_from_classes(Glm)
    def __init__(self,
                 pen_val=1, l1_ratio=0.5,
                 pen_func='log',
                 pen_func_kws={},
                 init='default',
                 pertub_init='n_samples',
                 groups=None,
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None,
                 adpt_weights=None,  # TODO: see above
                 ): pass

    def _get_solve_kws(self):
        """
        solve_glm is called as solve_glm(X=X, y=y, **kws)
        """

        if self.ridge_weights is not None and self.tikhonov is not None:
            raise ValueError("Both ridge weigths and tikhonov"
                             "cannot both be provided")

        loss_func, loss_kws = self.get_loss_info()

        lasso_pen, ridge_pen = \
            lasso_and_ridge_from_enet(pen_val=self.pen_val,
                                      l1_ratio=self.l1_ratio)

        kws = {'loss_func': loss_func,
               'loss_kws': loss_kws,

               'fit_intercept': self.fit_intercept,
               **self.opt_kws,

               'lasso_pen': lasso_pen,
               'ridge_pen': ridge_pen,

               }

        ###################################
        # potential extra Lasso arguments #
        ###################################

        # let's only add these if they are not None
        # this way we can use solvers that doesn't have these kws
        extra_kws = {
                     'ridge_weights': self.ridge_weights,
                     'tikhonov': self.tikhonov
                     }

        kws = maybe_add(kws, **extra_kws)

        ##################################
        # potential lasso type arguments #
        ##################################

        pen_kind = self._get_penalty_kind()
        if pen_kind == 'group':
            kws['groups'] = self.groups

        elif pen_kind == 'multi_task':
            kws['L1to2'] = True

        elif pen_kind == 'nuc':
            kws['nuc'] = True

        return kws

    def get_pen_val_max(self, X, y, init_data, sample_weight=None):
        l1_max = self._get_pen_max_lasso(X, y, init_data,
                                         sample_weight=sample_weight)

        return l1_max / self.l1_ratio


class GlmAdaptiveENetCVPath(AdptCVMixin, ENetCVPathMixin, GlmCVWithInitENet):
    solve_glm_path = None

    _cv_descr = dedent("""
        Tunes the Adaptive ElasticNet penalty parameter and or the l1_ratio via cross-validation. Makes use of a path algorithm for computing the penalty value tuning path.
        """)

    def _get_solve_path_enet_base_kws(self):
        kws = self.estimator._get_solve_kws()
        del kws['lasso_pen']
        del kws['ridge_pen']

        if hasattr(self, 'adpt_weights_'):
            kws['lasso_weights'] = self.adpt_weights_
        else:
            raise ValueError("adpt_weights has not been set")

        return kws

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmAdaptiveENet)


class GlmAdaptiveENetCVGridSearch(AdptCVMixin, CVGridSearchMixin,
                                  GlmCVWithInitENet):

    _cv_descr = dedent("""
        Tunes the Adaptive ElasticNet penalty parameter and or the l1_ratio via cross-validation.
        """)

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmAdaptiveENet)
