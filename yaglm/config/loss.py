from yaglm.config.base_params import ManualTunerMixin
from yaglm.config.base_loss import LossConfig


class LinReg(LossConfig):
    """
    Linear regression.
    """
    name = 'lin_reg'
    _estimator_type = "regressor"
    is_exp_fam = True
    has_scale = True
    def __init__(self): pass


class L2Reg(LossConfig):
    """
    L2 loss regresion i.e. the loss function used by the square root lasso.
    """
    name = 'l2'
    _estimator_type = "regressor"
    is_exp_fam = False
    has_scale = False  # TODO: maybe?
    def __init__(self): pass


class Huber(ManualTunerMixin, LossConfig):
    """
    Huber regression.

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

    Parameters
    ----------
    knot: quantile
        Which quantile we want to regress on.
    """
    name = 'quantile'
    _estimator_type = "regressor"

    is_exp_fam = False
    has_scale = False

    def __init__(self, quantile=0.5):
        self.quantile = quantile


class Poisson(LossConfig):
    """
    Poisson regression.
    """
    name = 'poisson'
    _estimator_type = "regressor"

    is_exp_fam = True
    has_scale = False


class LogReg(LossConfig):
    """
    Logistic regression.

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

    def get_loss_kws(self):
        return {}  # solvers do not need to know about class_weight


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
    _non_func_params = ['class_weight']

    is_exp_fam = True
    has_scale = False

    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def get_loss_kws(self):
        return {}  # solvers do not need to know about class_weight


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
                'poisson': Poisson(),
                'log_reg': LogReg(),
                'multinomial': Multinomial()
                }

avail_losses = list(loss_str2obj.keys())
