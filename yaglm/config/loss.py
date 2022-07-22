from yaglm.config.base_params import ManualTunerMixin
from yaglm.config.base_loss import LossConfig


class LinReg(LossConfig):
    """
    Linear regression.

    Single response: L(y, z) = 0.5 (y - z)^2

    Multiples response: L(y, z) = 0.5 ||y - z||_2^2
    """
    name = 'lin_reg'
    _estimator_type = "regressor"
    is_exp_fam = True
    has_scale = True
    def __init__(self): pass


class L2Reg(LossConfig):
    """
    L2 loss regresion i.e. the loss function used by the square root lasso.

    L(Y, Z) = 1/(sqrt(n)) ||Y - Z||_2
    Note this is not a separable loss!
    """
    name = 'l2'
    _estimator_type = "regressor"
    is_exp_fam = False
    has_scale = False  # TODO: maybe?
    def __init__(self): pass


class Huber(ManualTunerMixin, LossConfig):
    """
    Huber regression.

    L(y, z) = huber(y - z; knot)

    huber(r; z) = 0.5 * r^2 if |r| <= knot
                   knot * (|r| - 0.5 * knot) if |r| >= knot

    Parameters
    ----------
    knot: float
        Location of the knot for the huber function.
    """
    name = 'huber'
    _estimator_type = "regressor"
    _tunable_params = ['knot']
    is_exp_fam = False
    has_scale = False  # TODO: maybe?

    def __init__(self, knot=1.35):
        self.knot = knot


class Quantile(LossConfig):
    """
    Quantile regression.

    L(y, z) = tilted_L1(y - z; quant)

    tilted_L1(r; quant) = quant * r        if u >= 0
                          (quant - 1) * r  if r < 0

    Parameters
    ----------
    quantile: float
        Which quantile we want to regress on. Must be between 0 and 1.
    """
    name = 'quantile'
    _estimator_type = "regressor"

    is_exp_fam = False
    has_scale = False

    def __init__(self, quantile=0.5):
        self.quantile = quantile


class SmoothedQuantile(LossConfig):
    """
    Quantile regression.

    L(y, z) = smooth-tilted_L1(y - z; quant, smooth-param)

    smooth-tilted_L1(r; q, a) = q * r + q * log(1 + e^{-r/a})

    Parameters
    ----------
    quantile: float
        Which quantile we want to regress on. Must be between 0 and 1.

    smooth_param: float
        The smoothing parameters. Must be non-negative. Smaller values make this closer to the tilted L1 loss, but less smooth.

    References
    ----------
    Zheng, S., 2011. Gradient descent algorithms for quantile regression with smooth approximation. International Journal of Machine Learning and Cybernetics, 2(3), pp.191-207.
    """
    name = 'smoothed_quantile'
    _estimator_type = "regressor"

    is_exp_fam = False
    has_scale = False

    def __init__(self, quantile=0.5, smooth_param=0.5):
        self.quantile = quantile
        self.smooth_param = smooth_param


class Poisson(LossConfig):
    """
    Poisson regression.
    L(y, z) = exp(z) - z * y
    """
    name = 'poisson'
    _estimator_type = "regressor"

    is_exp_fam = True
    has_scale = False


class LogReg(LossConfig):
    """
    Logistic regression.

    Assumes the y responses are binary!

    L(y, z) = - log(s(z))       if y == 1
              - log(1 - s(z))   if y == 0

    where s(z) = 1 / (1 + e^-z) is the sigmoid function

    Parameters
    ----------
    class_weight : dict or 'balanced', default=None

        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    """
    name = 'log_reg'
    _estimator_type = "classifier"
    _non_func_params = ['class_weight']

    is_exp_fam = True
    has_scale = False

    def __init__(self, class_weight=None):
        self.class_weight = class_weight


class Multinomial(LossConfig):
    """
    Multinomial regression.

    Parameters
    ----------
    class_weight : dict or 'balanced', default=None

        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    """
    name = 'multinomial'
    _estimator_type = "classifier"
    _non_func_params = ['clas_weight']

    is_exp_fam = True
    has_scale = False

    def __init__(self, class_weight=None):
        self.class_weight = class_weight


class Hinge(LossConfig):
    """
    Hindge loss funciton

    Assumes the y responses are +/- 1.

    L(y, z) = max(0, 1 - y * z) = hinge(y * z)
    where hinge(p) = max(0, 1 - p)
    """
    name = 'hinge'
    _estimator_type = "classifier"

    is_exp_fam = False
    has_scale = False
    def __init__(self): pass


class HuberizedHinge(LossConfig):
    """
    The Huberized hinge loss funciton

    Assumes the y responses are +/- 1.

    L(y, z) = huberized-hinge(y * z)

    huberized-hinge(p) = 0                 if p >= 1
                         0.5 * (1 - p)^2   if 0 < p < 1
                         0.5 - p           if p <= 0

    TODO: maybe add knot parameter

    References
    ----------
    Rennie, J.D. and Srebro, N., 2005, July. Loss functions for preference levels: Regression with discrete ordered labels. In Proceedings of the IJCAI multidisciplinary workshop on advances in preference handling (Vol. 1). AAAI Press, Menlo Park, CA.
    """
    name = 'huberized_hinge'
    _estimator_type = "classifier"

    is_exp_fam = False
    has_scale = False

    def __init__(self): pass


class LogisticHinge(LossConfig):
    """
    The logistic regresssion loss where the responses are +/-1 i.e. a smooth stand in for the hinge loss.


    L(y, z) = logistic(y * z)

    logistic(p) = log(1 + e^-p)

    # TODO: add class weight!
    """
    name = 'logistic_hinge'
    _estimator_type = "classifier"

    is_exp_fam = True
    has_scale = False

    def __init__(self): pass


def get_loss_config(config):
    """
    Gets the loss config object.

    Parameters
    ----------
    loss: str, LossConfig, TunerConfig or None
        The loss.

    Output
    ------
    config: LossConfig, None
        The loss config object.
    """
    if type(config) != str:
        return config
    else:

        return loss_str2obj[config.lower()]


loss_str2obj = {'lin_reg': LinReg(),
                'l2': L2Reg(),
                'huber': Huber(),

                'quantile': Quantile(),
                'smooth_quantile': SmoothedQuantile(),

                'poisson': Poisson(),

                'log_reg': LogReg(),
                'multinomial': Multinomial(),

                'hinge': Hinge(),
                'huberized_hinge': HuberizedHinge(),
                'logistic_hinge': LogisticHinge()
                }

avail_losses = list(loss_str2obj.keys())
