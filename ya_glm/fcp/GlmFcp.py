from copy import deepcopy
import numpy as np
from scipy.sparse import diags
from textwrap import dedent

from ya_glm.Glm import Glm
from ya_glm.autoassign import autoassign
from ya_glm.fcp.utils import fit_if_unfitted
from ya_glm.fcp.fcp_pen_max import get_fcp_pen_val_max
from ya_glm.utils import is_multi_response, get_coef_and_intercept
from ya_glm.opt.concave_penalty import get_penalty_func


class InitMixin:
    """
    _get_defualt_init

    _get_init_data_from_fit_est
    """

    def get_init_data(self, X, y=None):
        """
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

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
                default_init = fit_if_unfitted(defualt_init, X, y)
                return self._get_init_data_from_fit_est(default_init)

        else:

            # user provided an estimator
            init_est = fit_if_unfitted(self.init, X, y)
            return self._get_init_data_from_fit_est(init_est)

    def _get_defualt_init(self):
        raise NotImplementedError

    def _get_init_data_from_fit_est(self, est):
        raise NotImplementedError


# _glm_fcp_base_params = dedent("""
#     pen_val: float
#         The penalty parameter value.

#     pen_func: str
#         Which concave penalty function to use. See TODO for list of supported penalty functions.

#     pen_func_kws: dict
#         Key word arguments for the penalty function parameters. E.g. to set the 'a' parameter for the SCAD penalty this would be pen_func_kws = {'a': 3.7}

#     init: str, dict, Estimator
#         Where to initialize the optimization algorithm. If 'defualt' then a defualt initialization strategy will be used (e.g. LassoCV). If a dict with keys 'coef' and 'intercept' is provided to specify a particular initial point. If an estimator is provided then the initial values will be extracted from the estimator. If the estimator is not yet fitted it will be first fitted to the training data.

#     fit_intercept: bool
#         Whether or not to fit an intercept.

#     opt_kws: dict
#         Key word arguments to the optimization algorithm.
#     """)


class GlmFcp(Glm, InitMixin):

    @autoassign
    def __init__(self,
                 fit_intercept=True,
                 standardize=False,
                 opt_kws={},

                 pen_val=1,
                 pen_func='scad',
                 pen_func_kws={},
                 init='default',
                 ):
        pass

    def fit(self, X, y):

        # validate the data!
        X, y = self._validate_data(X, y)

        # get data for initialization
        init_data = self.get_init_data(X, y)
        if 'est' in init_data:
            self.init_est_ = init_data['est']

        # pre-process data
        X, y, pre_pro_out = self._pre_process(X, y, copy=True)

        # possibly process the init data e.g. shift/scale
        init_data_pro = self._process_init_data(init_data, pre_pro_out)

        # Fit!
        fit_data = self._compute_fit(X, y, init_data=init_data_pro)

        self._set_fit(fit_data, pre_pro_out)
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

    def _compute_fit(self, X, y, init_data):
        raise NotImplementedError

    def _process_init_data(self, init_data, pre_pro_out):
        """
        Process the initialization data. E.g. if we center/scale our X data before fitting we should transform the coef/intercept initializers to match the center/scaled data.

        This should not modify init_data

        Parameters
        ----------
        init_data: dict

        pre_pro_out: dict

        Output
        ------
        init_data_pro: dict
        """
        # TODO: double check this

        init_data_pro = {}

        # coefficient
        coef = np.array(init_data['coef'])
        is_mr = is_multi_response(coef)
        if not is_mr:
            coef = coef.ravel()

        # rescale coefficient
        if pre_pro_out is not None and 'X_scale' in pre_pro_out:
            coef = diags(pre_pro_out['X_scale']) @ coef

        init_data_pro['coef'] = coef

        # intercept
        if 'intercept' in init_data:
            intercept = deepcopy(init_data['intercept'])

            if intercept is not None:
                if pre_pro_out is not None and 'X_offset' in pre_pro_out:
                    intercept += coef.T @ pre_pro_out['X_offset']

                if pre_pro_out is not None and 'y_offset' in pre_pro_out:
                    intercept -= pre_pro_out['y_offset']

            init_data_pro['intercept'] = intercept

        return init_data_pro

    def get_pen_val_max(self, X, y, init_data):
        X_pro, y_pro, pre_pro_out = self._pre_process(X, y, copy=True)
        init_data_pro = self._process_init_data(init_data, pre_pro_out)
        return self._get_pen_val_max_from_pro(X=X_pro, y=y_pro,
                                              init_data=init_data_pro)

    def _get_pen_val_max_from_pro(self, X, y, init_data):

        return get_fcp_pen_val_max(X=X, y=y, init_data=init_data,
                                   model_type=self._model_type,
                                   pen_func=self.pen_func,
                                   pen_func_kws=self.pen_func_kws,
                                   fit_intercept=self.fit_intercept)


# GlmFcp.__doc__ = dedent("""
#     Generalized linear model penalized by a folded concave penalty.

#     Parameters
#     ----------
#     {}
#     """.format(_glm_fcp_base_params)
# )


# _lla_params = dedent("""
# lla_n_steps: int
#     Maximum number of LLA steps to take.

# lla_kws: dict
#     Key word arguments to LLA algorithm. See TODO.
# """)


class GlmFcpFitLLA(GlmFcp):

    base_wl1_solver = None
    solve_lla = None

    # HACK to get around inheretance signature issue
    # TODO: remove init and pen_val after figuring out signature issue
    # @autoassign
    def __init__(self, pen_val=1,
                 lla_n_steps=1, lla_kws={},
                 init='default',
                 **kws):
        super().__init__(**kws)
        self.pen_val = pen_val
        self.lla_n_steps = lla_n_steps
        self.lla_kws = lla_kws
        self.init = init

    def _compute_fit(self, X, y, init_data):

        coef_init = init_data['coef']
        if self.fit_intercept:
            intercept_init = init_data['intercept']
        else:
            intercept_init = None

        # Setup penalty function
        penalty_func = get_penalty_func(pen_func=self.pen_func,
                                        pen_val=self.pen_val,
                                        pen_func_kws=self.pen_func_kws)

        # setup solver for weighted Lasso solver
        wl1_solver = self.base_wl1_solver(X=X, y=y,
                                          **self._get_wl1_solver_kws())

        # solve!
        coef, intercept, opt_data = \
            self.solve_lla(wl1_solver=wl1_solver,
                           penalty_fcn=penalty_func,
                           init=coef_init,
                           init_upv=intercept_init,
                           n_steps=self.lla_n_steps,
                           **self.lla_kws)

        return {'coef': coef, 'intercept': intercept, 'opt_data': opt_data}

    def _get_wl1_solver_kws(self):
        # raise NotImplementedError
        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                'opt_kws': self.opt_kws}


# GlmFcpFitLLA.__doc__ = dedent("""
#     Generalized linear model penalized by a folded concave penalty and fit with the local linear approximation (LLA) algorithm.

#     Parameters
#     ----------
#     {}

#     {}
#     """.format(_glm_fcp_base_params, _lla_params)
# )
