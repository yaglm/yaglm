from textwrap import dedent

from ya_glm.base.Glm import PenGlm
from ya_glm.base.LossMixin import LossMixin

from ya_glm.base.GlmWithInit import GlmWithInitMixin
from ya_glm.base.GlmCVWithInit import GlmCVWithInitSinglePen

from ya_glm.pen_max.fcp_lla import get_pen_max
from ya_glm.lla.lla import solve_lla

from ya_glm.opt.penalty.concave_penalty import get_penalty_func
from ya_glm.init_signature import add_from_classes
from ya_glm.utils import maybe_add
from ya_glm.processing import check_estimator_type, process_init_data
from ya_glm.make_docs import merge_param_docs

_fcp_lla__params = dedent("""
pen_val: float
    The penalty value for the concave penalty.

pen_func: str
    The concave penalty function.

pen_func_kws: dict
    Keyword arguments for the concave penalty function e.g. 'a' for the SCAD function.

init: str, estimator, dict with key ['coef', 'intercept'].
    Where the LLA algorithm is initialized from. If a fitted estimator or dict is provided the initial values will be exctracted. If an unfitted estimator is provided the estimator will be fitted from the data. If 'default' then a default initializer will be used.

lla_n_steps: int
    Number of LLA steps to take from the initializer.

lla_kws: dict
    Additional keyword arguments to self.solve_lla.

groups: None, list of ints
    Optional groups of variables. If groups is provided then each element in the list should be a list of feature indices. Variables not in a group are not penalized.

multi_task: bool
    Use multi-task lasso on the coefficient matrix for multiple response cases.

nuc: bool
    Apply the penalty to the singular values of the coefficient matrix for multiple response cases.

ridge_pen_val: None, float
    Penalty strength for an optional ridge penalty.

ridge_weights: None, array-like shape (n_featuers, )
    Optional features weights for the ridge peanlty.

tikhonov: None, array-like (K, n_features)
    Optional tikhonov matrix for the ridge penalty. Both tikhonov and ridge weights cannot be provided at the same time.
    """)


class GlmFcpLLA(LossMixin, GlmWithInitMixin, PenGlm):
    solve_lla = staticmethod(solve_lla)
    solve_glm = None
    WL1Solver = None  # base class for WL1 solver

    _pen_descr = dedent("""
    Folded concave penalty fit by applying the local linear approximation (LLA) algorithm to a good initializer.
    """)

    _params_descr = merge_param_docs(_fcp_lla__params, PenGlm._params_descr)

    @add_from_classes(PenGlm)
    def __init__(self,
                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 init='default',
                 lla_n_steps=1, lla_kws={},
                 groups=None,
                 multi_task=False,
                 nuc=False,
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None
                 ): pass

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

        # extract coef / intercept init
        coef_init = init_data_pro['coef']
        if self.fit_intercept and 'intercept' in init_data_pro:
            intercept_init = init_data_pro['intercept']
        else:
            intercept_init = None

        # Setup penalty function
        penalty_func = get_penalty_func(pen_func=self.pen_func,
                                        pen_val=self.pen_val,
                                        pen_func_kws=self.pen_func_kws)

        # Setup weighted L1 solver
        wl1_solver = self.WL1Solver(X=X, y=y,
                                    loss_func=self.loss_func,
                                    loss_kws=self.get_loss_kws(),
                                    fit_intercept=self.fit_intercept,
                                    sample_weight=sample_weight,
                                    solver_kws=self._get_extra_solver_kws())

        wl1_solver.solve_glm = self.solve_glm

        ###############################
        # Fit with the LLA algorithm! #
        ###############################
        coef, intercept, opt_data = \
            self.solve_lla(wl1_solver=wl1_solver,
                           penalty_fcn=penalty_func,
                           init=coef_init,
                           init_upv=intercept_init,
                           transform=self._get_coef_transform(),
                           n_steps=self.lla_n_steps,
                           **self.lla_kws)

        # set the fit
        fit_out = {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}
        self._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
        return self

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

        pen_kind = self._get_penalty_kind()
        if pen_kind == 'group':
            groups = self.groups
        else:
            groups = None

        return get_pen_max(X=X, y=y, init_data=init_data,
                           pen_func=self.pen_func,
                           pen_func_kws=self.pen_func_kws,
                           loss_func=self.loss_func,
                           loss_kws=self.get_loss_kws(),
                           groups=groups,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight,
                           pen_kind=pen_kind)

    def _get_extra_solver_kws(self):

        if self.ridge_weights is not None and self.tikhonov is not None:
            raise ValueError("Both ridge weigths and tikhonov"
                             "cannot both be provided")

        kws = {**self.opt_kws}

        ###################################
        # potential extra Lasso arguments #
        ###################################

        # let's only add these if they are not None
        # this way we can use solvers that doesn't have these kws
        extra_kws = {'ridge_pen': self.ridge_pen_val,
                     'ridge_weights': self.ridge_weights,
                     'tikhonov': self.tikhonov,
                     'groups': self.groups,
                     'multi_task': self.multi_task,
                     'nuc': self.nuc
                     }

        kws = maybe_add(kws, **extra_kws)

        return kws

    def _kws_for_default_init(self, c=None):

        keys = ['fit_intercept', 'standardize', 'opt_kws',
                'ridge_pen_val', 'ridge_weights', 'tikhonov',
                'groups', 'multi_task', 'nuc']

        return {k: self.__dict__[k] for k in keys}


class GlmFcpLLACV(GlmCVWithInitSinglePen):

    @property
    def has_path_algo(self):
        return False

    _cv_descr = dedent("""
    Tunes the penalty parameter of a concave penalty function fit with the LLA algorithm via cross-validation.
    """)

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmFcpLLA)
