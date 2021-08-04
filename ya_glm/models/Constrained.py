from ya_glm.base.Glm import Glm
from ya_glm.ConstraintConfig import ConstraintConfig
from ya_glm.loss.LossMixin import LossMixin
from ya_glm.autoassign import autoassign


class Constrained(LossMixin, Glm):
    """
    Constrained GLM.

    Parameters
    ----------
    loss: str, ya_glm.LossConfig.LossConfig
        The loss function. If a string is provided the loss function parameters are set to their default values. Otherwise the loss function parameters can be specified by providing a LossConfig object. See ya_glm.LossConfig for available loss functions.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    pos: bool
        Constrain the coefficeint to be positive.

    standardize: bool
        Whether or not to perform internal standardization before fitting the data. Standardization means mean centering and scaling each column by its standard deviation. For the group lasso penalty an additional scaling is applied that scales each variable by 1 / sqrt(group size). Putting each variable on the same scale makes sense for fitting penalized models. Note the fitted coefficient/intercept is transformed to be on the original scale of the input data.

    solver: str, ya_glm.GlmSolver
        The solver used to solve the penalized GLM optimization problem. If this is set to 'default' we try to guess the best solver. Otherwise a custom solver can be provided by specifying a GlmSolver object.

    Attributes
    ----------
    coef_: array-like, shape (n_features, ) or (n_features, n_responses)
        The fitted coefficient vector or matrix (for multiple responses).

    intercept_: None, float or array-like, shape (n_features, )
        The fitted intercept.

    classes_: array-like, shape (n_classes, )
        A list of class labels known to the classifier.

    opt_data_: dict
        Data output by the optimization algorithm.
    """
    @autoassign
    def __init__(self, loss='lin_reg', fit_intercept=True,
                 pos=False,
                 standardize=True, solver='default'): pass

    def _get_penalty_config(self):
        return ConstraintConfig(pos=self.pos)
