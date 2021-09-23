from copy import deepcopy

from ya_glm.config.base import Config, ManualTunerMixin, safe_get_config
from ya_glm.autoassign import autoassign
from ya_glm.utils import is_str_and_matches, delete_fit_attrs


class FlavorConfig(Config):
    """
    Base penalty flavor config object.
    """

    def get_default_init(self, est):
        """
        Gets the default initializer; this is simply the unflavored original estimator.

        Parameters
        ----------
        est: Estimator
            The original estimator object.

        Output
        ------
        default: Estimator
            The unflavored version of the original object.
        """
        if est._is_tuner:
            default = deepcopy(est)

            default = delete_fit_attrs(default)  # remove fit attributes
            # unflavor the penalty
            default.penalty.flavor = None
        else:

            # importing this here avoids circular import issues
            from ya_glm.GlmTuned import GlmCV

            params = deepcopy(est.get_params(deep=False))
            # unflavor the penalty
            params['penalty'].flavor = None
            default = GlmCV(**params)

        return default

    def get_initialized_penalty(self, penalty, init_data):
        """
        Returns the initialized penalty config e.g. sets adaptive weights.
        """
        raise NotImplementedError

    def transform_pen_max(self, cvx_max_val):
        """
        After computing the pen_val_max for the corresponding convex problem this (possibly) returns the pen_val_max for the the flavored problem.

        Parameters
        ----------
        cvx_max_val: float
            The largest reasonable penalty value for the convex problem.

        Output
        ------
        flavored_max_val: float
            The largest reasonable penalty value for the flavored problem.
        """
        # TODO-THINK-THROUGH: The logic of this may be a bit subtle
        raise NotImplementedError("Subclass should overwrite")

    @property
    def needs_separate_solver_init(self):
        """
        TODO: document
        """
        return False


class Adaptive(ManualTunerMixin, FlavorConfig):
    """
    An adaptive penalty.

    Parameters
    ----------
    expon: float
        A positive number indicating the exponent for the adpative weights.
        The adpative weights are set via 1 / (abs(init_coef) + pertub_init) ** expon

    pertub_init: str, float
        How to perturb the initial coefficient before computing the adpative weights. This perturbation is useful  when the init coefficient has exact zeros. If pertub_init='n_samples', then we use 1/n_samples.

    Attributes
    ----------
    adpt_weights_: array-like
        The adpative weights.

    References
    ----------
    Zou, H., 2006. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476), pp.1418-1429.
    """
    name = 'adaptive'
    _tunable_params = ['expon']

    @autoassign
    def __init__(self, expon=1, pertub_init='n_samples'): pass

    def get_initialized_penalty(self, penalty, init_data):
        """
        Returns the penalty config with the set adaptive weights.

        Parameters
        ----------
        penalty: PenaltyConfig
            The original penalty config.

        init_data: dict
            The pre-processed initializer data. Must have keys 'coef' and 'n_samples'.

        Output
        ------
        set_penalty: PenaltyConfig
            The penalty with the adaptive weights set.
        """

        # possibly transform coef_init
        transform = penalty.get_non_smooth_transforms()
        if transform is None:
            transform = abs
        coef_transf = transform(init_data['coef'])

        # possibly perturb the transformed coefficient
        if is_str_and_matches(self.pertub_init, 'n_samples'):

            coef_transf += 1 / init_data['n_samples']
        elif self.pertub_init is not None:
            coef_transf += self.pertub_init

        # set adpative weights
        adpt_weights = 1 / (coef_transf ** self.expon)

        # set the weights for the penalty.
        penalty.set_params(weights=adpt_weights)

        return penalty

    def transform_pen_max(self, cvx_max_val, penalty):
        """
        After computing the pen_val_max for the corresponding convex problem this (possibly) returns the pen_val_max for the the flavored problem.

        Parameters
        ----------
        cvx_max_val: float
            The largest reasonable penalty value for the convex problem.


        penalty: PenaltyConfig
            The penalty config object.

        Output
        ------
        flavored_max_val: float
            The largest reasonable penalty value for the flavored problem.
        """
        # this value should have been already computed with the adaptive weights
        return cvx_max_val


class NonConvexDirect(ManualTunerMixin, FlavorConfig):
    """
    A non-convex penalty fit directly.

    Parameters
    ----------
    pen_func: str
        The concave penalty function. See ya_glm.opt.penalty.concave_penalty.

    second_param_val: float, str
        Value of the secondary penalty parameter value for the non-convex penalty e.g. 'a' for SCAD, 'q' for the Lq norm. If None, will use a sensible default e.g. a=3.7 for SCAD.
    """
    name = 'non_convex_direct'
    _tunable_params = ['second_param_val']

    @autoassign
    def __init__(self, pen_func='scad', second_param_val=None): pass

    def get_initialized_penalty(self, penalty, init_data):
        """
        Returns the penalty with the coefficient initializer set.

        Parameters
        ----------
        penalty: PenaltyConfig
            The original penalty config.

        init_data: dict
            The pre-processed initializer data. Must have keys 'coef' and 'n_samples'.

        Output
        ------
        set_penalty: PenaltyConfig
            The penalty with set initializer.
        """
        penalty.coef_init_ = init_data['coef']
        penalty.intercept_init_ = init_data['intercept']
        return penalty

    def get_default_init(self, est):
        """
        By default we initialize from zero.
        """
        return 'zero'

    def transform_pen_max(self, cvx_max_val, penalty):
        """
        After computing the pen_val_max for the corresponding convex problem this (possibly) returns the pen_val_max for the the flavored problem.

        Parameters
        ----------
        cvx_max_val: float
            The largest reasonable penalty value for the convex problem.


        penalty: PenaltyConfig
            The penalty config object.

        Output
        ------
        flavored_max_val: float
            The largest reasonable penalty value for the flavored problem.
        """
        if self.pen_func in ['mcp', 'scad']:
            return cvx_max_val
        else:
            raise NotImplementedError("TODO add")

    @property
    def needs_separate_solver_init(self):
        """
        TODO: document
        """
        return True


class NonConvexLLA(ManualTunerMixin, FlavorConfig):
    """
    A non-convex penalty fit with the LLA algorithm.

    Parameters
    ----------
    pen_func: str
        The concave penalty function. See ya_glm.opt.penalty.concave_penalty.

    second_param_val: None, float
        Value of the secondary penalty parameter value for the non-convex penalty e.g. 'a' for SCAD, 'q' for the Lq norm. If None, will use a sensible default e.g. a=3.7 for SCAD.

    lla_n_steps: int
        Maximum of steps the LLA algorithm should take. The LLA algorithm can have favorable statistical properties after only 1 step.

    lla_kws: dict
        Additional keyword arguments to the LLA algorithm solver excluding 'n_steps' and 'glm_solver'. See ya_glm.lla.LLASolver.LLASolver.

    References
    ----------
    Zou, H. and Li, R., 2008. One-step sparse estimates in nonconcave penalized likelihood models. Annals of statistics, 36(4), p.1509.

    Fan, J., Xue, L. and Zou, H., 2014. Strong oracle optimality of folded concave penalized estimation. Annals of statistics, 42(3), p.819.
    """

    name = 'non_convex_lla'
    _tunable_params = ['second_param_val']

    @autoassign
    def __init__(self, pen_func='scad', second_param_val=None,
                 lla_n_steps=1, lla_kws={}): pass

    def get_initialized_penalty(self, penalty, init_data):
        """
        Returns the penalty with the coefficient initializer set.

        Parameters
        ----------
        penalty: PenaltyConfig
            The original penalty config.

        init_data: dict
            The pre-processed initializer data. Must have keys 'coef' and 'n_samples'.

        Output
        ------
        set_penalty: PenaltyConfig
            The penalty with set initializer.
        """
        penalty.coef_init_ = init_data['coef']
        penalty.intercept_init_ = init_data['intercept']
        return penalty

    @property
    def lla_solver_kws(self):
        kws = {'n_steps': self.lla_n_steps,
               **self.lla_kws}

        # TODO: maybe include an option for something like this?
        # if self.lla_n_steps <= 2:
        # kws['xtol'] = 1e-4
        # kws['atol'] = None
        # kws['rtol'] = None

        return kws

    def transform_pen_max(self, cvx_max_val, penalty):
        """
        After computing the pen_val_max for the corresponding convex problem this (possibly) returns the pen_val_max for the the flavored problem.

        Parameters
        ----------
        cvx_max_val: float
            The largest reasonable penalty value for the convex problem.

        penalty: PenaltyConfig
            The penalty config object.

        Output
        ------
        flavored_max_val: float
            The largest reasonable penalty value for the flavored problem.
        """

        transf = penalty.get_non_smooth_transforms()
        pen_func = penalty.get_transf_nonconex_penalty()
        coef_init = penalty.coef_init_

        if not hasattr(pen_func, 'fcp_data'):
            raise NotImplementedError("TODO Need to add FCP data for"
                                      "this penalty")
        a1 = pen_func.fcp_data['a1']
        b1 = pen_func.fcp_data['b1']

        return max(abs(transf(coef_init)).max() / b1, cvx_max_val / a1)


def get_flavor_config(config):
    """
    Gets the flavor config object. If a tuned flavor is provided this will return the base flavor config.

    Parameters
    ----------
    config: str, FlavorConfig or TunerConfig
        The penalty flavor.

    Output
    ------
    config: FlavorConfig:
        The flavor config object.
    """
    if type(config) != str:
        return safe_get_config(config)
    else:
        return flavor_str2obj[config.lower()]


flavor_str2obj = {'adaptive': Adaptive(),
                  'non_convex_lla': NonConvexLLA(),
                  'non_convex_direct': NonConvexDirect()
                  }

avail_flavors = list(flavor_str2obj.keys())
