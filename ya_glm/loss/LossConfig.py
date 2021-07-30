from copy import deepcopy


class LossConfig:
    name = None
    _estimator_type = None

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
    balence_classes: bool
        Whether or not we want to balance classes by class sizes.
    """
    name = 'log_reg'
    _estimator_type = "classifier"

    def __init__(self, balence_classes=False):
        self.balence_classes = balence_classes


class Multinomial(LossConfig):
    """
    Multinomial regression.

    Parameters
    ----------
    balence_classes: bool
        Whether or not we want to balance classes by class sizes.
    """
    name = 'multinomial'
    _estimator_type = "classifier"

    def __init__(self, balence_classes=False):  # TODO: doc from sklearn
        self.balence_classes = balence_classes


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
