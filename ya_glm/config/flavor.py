from ya_glm.config.base_params import ParamConfig, ManualTunerMixin
from ya_glm.autoassign import autoassign

# TODO: maybe add proximable tags?
class FlavorConfig(ParamConfig):
    """
    Base penalty flavor config object.

    Class attributes
    ----------------
    name: str
    """
    pass


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

    References
    ----------
    Zou, H., 2006. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476), pp.1418-1429.
    """
    name = 'adaptive'
    _tunable_params = ['expon']

    @autoassign
    def __init__(self, expon=1, pertub_init='n_samples'): pass


class NonConvex(ManualTunerMixin, FlavorConfig):
    """
    A non-convex penalty fit directly.

    Parameters
    ----------
    pen_func: str
        The concave penalty function. See ya_glm.opt.penalty.concave_penalty.

    second_param_val: float, str
        Value of the secondary penalty parameter value for the non-convex penalty e.g. 'a' for SCAD, 'q' for the Lq norm. If None, will use a sensible default e.g. a=3.7 for SCAD.
    """
    name = 'non_convex'
    _tunable_params = ['second_param_val']

    @autoassign
    def __init__(self, pen_func='scad', second_param_val=None): pass


def get_flavor_config(config):
    """
    Gets the flavor config object.

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
        return config
    else:
        return flavor_str2obj[config.lower()]


flavor_str2obj = {'adaptive': Adaptive(),
                  'non_convex': NonConvex(),
                  }

avail_flavors = list(flavor_str2obj.keys())
