from copy import deepcopy


class LossConfig:
    name = None
    _estimator_type = None

    # TODO: should we rename this solve_kws()
    @property
    def loss_kws(self):
        return deepcopy(self.__dict__)


class LinReg(LossConfig):
    """
    Linear regression.
    """
    name = 'lin_reg'
    _estimator_type = "regressor"


class Huber(LossConfig):
    """
    Huber regression.

    Parameters
    ----------
    knot: float
        Location of the knot for the huber function.
    """
    name = 'huber'
    _estimator_type = "regressor"

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

    def __init__(self, quantile=0.5):
        self.quantile = quantile


class Poisson(LossConfig):
    """
    Poisson regression.
    """
    name = 'poisson'
    _estimator_type = "regressor"


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

    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    @property
    def loss_kws(self):
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

    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    @property
    def loss_kws(self):
        return {}  # solvers do not need to know about class_weight


def get_loss_config(loss):
    """
    Parameters
    ----------
    loss: str or a loss config object

    Output
    ------
    loss_config:
        A loss config object
    """
    if type(loss) != str:
        return loss
    else:
        return loss_str_to_default[loss.lower()]


loss_str_to_default = {'lin_reg': LinReg(),
                       'huber': Huber(),
                       'quantile': Quantile(),
                       'poisson': Poisson(),
                       'log_reg': LogReg(),
                       'multinomial': Multinomial()
                       }

avail_losses = list(loss_str_to_default.keys())
