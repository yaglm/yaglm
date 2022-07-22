from yaglm.config.base_params import ParamConfig


class LossConfig(ParamConfig):
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
