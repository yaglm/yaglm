from ya_glm.config.base import Config, ManualTunerMixin, safe_get_config


class LossConfig(Config):
    """
    Base GLM loss config objecct.

    Class Attributes
    ---------------
    name: str
        Name of this loss function.

    _estimator_type: str
        Estimator type e.g. must be on of ['classifier', 'regressor']

    _non_func_params: list of str
        Any parameters that do not apply to the loss function e.g. class_weight for logistic regression

    is_exp_fam: bool
        Whether or not this is actually a generalized linear model coming from an exponential family (e.g. linear, logistic poisson). False for things like quantile regression.

    has_scale: bool
        Whether or not this GLM family requires a scale parameter.

    """
    name = None
    _estimator_type = None
    _non_func_params = []
    is_exp_fam = False
    has_scale = False

    def get_func_params(self):
        """
        Returns the dict of parameters that describe the loss function.

        Output
        ------
        params: dict
            The loss function parameters.
        """

        all_params = self.get_params(deep=False)

        if self._non_func_params is not None \
                and len(self._non_func_params) > 0:
            to_drop = set(self._non_func_params)
            return {k: v for (k, v) in all_params.items() if k not in to_drop}
        else:
            return all_params


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
    Gets the loss config object. If a tuned loss is provided this will return the base loss config.

    Parameters
    ----------
    loss: str, LossConfig, or TunerConfig
        The loss.

    Output
    ------
    config: LossConfig:
        The loss config object.
    """
    if type(config) != str:
        return safe_get_config(config)
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
