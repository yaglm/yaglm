# from textwrap import dedent

from ya_glm.base.Glm import Glm
from ya_glm.base.GlmWithInit import GlmWithInitMixin
from ya_glm.base.GlmCVWithInit import GlmCVWithInitSinglePen
from ya_glm.cv.CVGridSearch import CVGridSearchMixin

from ya_glm.pen_max.fcp_lla import get_pen_max
from ya_glm.lla.lla import solve_lla

from ya_glm.opt.concave_penalty import get_penalty_func
from ya_glm.init_signature import add_from_classes, keep_agreeable
from ya_glm.utils import maybe_add
from ya_glm.processing import check_estimator_type

# _lla_params = dedent("""
# lla_n_steps: int
#     Maximum number of LLA steps to take.

# lla_kws: dict
#     Key word arguments to LLA algorithm. See TODO.
# """)


class GlmFcpLLA(GlmWithInitMixin, Glm):
    solve_lla = staticmethod(solve_lla)
    solve_glm = None
    WL1Solver = None  # base class for WL1 solver

    @add_from_classes(Glm)
    def __init__(self,
                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 init='default',
                 lla_n_steps=1, lla_kws={},
                 ridge_pen_val=None, ridge_weights=None, tikhonov=None,
                 groups=None,
                 ): pass

    def compute_fit(self, X, y, init_data):

        coef_init = init_data['coef']
        if self.fit_intercept:
            intercept_init = init_data['intercept']
        else:
            intercept_init = None

        # Setup penalty function
        penalty_func = get_penalty_func(pen_func=self.pen_func,
                                        pen_val=self.pen_val,
                                        pen_func_kws=self.pen_func_kws)

        # Setup weighted L1 solver
        loss_func, loss_kws = self.get_loss_info()

        wl1_solver = self.WL1Solver(X=X, y=y,
                                    loss_func=loss_func,
                                    loss_kws=loss_kws,
                                    fit_intercept=self.fit_intercept,
                                    solver_kws=self._get_extra_solver_kws())

        wl1_solver.solve_glm = self.solve_glm

        # solve!
        coef, intercept, opt_data = \
            self.solve_lla(wl1_solver=wl1_solver,
                           penalty_fcn=penalty_func,
                           init=coef_init,
                           init_upv=intercept_init,
                           transform=self._get_coef_transform(),
                           n_steps=self.lla_n_steps,
                           **self.lla_kws)

        return {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}

    def _get_pen_val_max_from_pro(self, X, y, init_data):

        loss_func, loss_kws = self.get_loss_info()
        pen_kind = self._get_penalty_kind()
        if pen_kind == 'group':
            groups = self.groups
        else:
            groups = None

        return get_pen_max(X=X, y=y, init_data=init_data,
                           pen_func=self.pen_func,
                           pen_func_kws=self.pen_func_kws,
                           loss_func=loss_func,
                           loss_kws=loss_kws,
                           groups=groups,
                           fit_intercept=self.fit_intercept,
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

    def _kws_for_default_init(self, c=None):

        keys = ['fit_intercept', 'standardize', 'opt_kws',
                'ridge_pen_val', 'ridge_weights', 'tikhonov',
                'groups']

        if hasattr(self, 'multi_task'):
            keys.append('multi_task')

        if hasattr(self, 'nuc'):
            keys.append('nuc')

        if c is not None:
            keys = keep_agreeable(keys, func=c.__init__)

        return {k: self.__dict__[k] for k in keys}


# GlmFcpFitLLA.__doc__ = dedent("""
#     Generalized linear model penalized by a folded concave penalty and fit with the local linear approximation (LLA) algorithm.

#     Parameters
#     ----------
#     {}

#     {}

#     {}
#     """.format(_glm_base_params, _glm_fcp_base_params, _lla_params)
# )
# TODO: better solution than manually adding all of these


class GlmFcpLLACV(CVGridSearchMixin, GlmCVWithInitSinglePen):

    def _check_base_estimator(self, estimator):
        check_estimator_type(estimator, GlmFcpLLA)
