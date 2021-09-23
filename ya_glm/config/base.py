from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from collections import defaultdict
# import re
# from sklearn import __version__
# import warnings

from ya_glm.autoassign import autoassign


class _Config:
    """
    Base configuration object e.g. for a GLM loss, penalty, penalty flavor, or constraint. This is very simlar to sklearn's BaseEstimator class e.g. most of the code is simply copied.
    """
    def _get_param_names(self):
        """
        Gets the names of all the parameters. Any attribute that does not end in '_' is considered a parameter.

        Output
        ------
        names: list of str
            The parameter names.
        """
        return [k for k in self.__dict__.keys() if k[-1] != '_']

    # TODO: do we need the class method version that does not allow arg/keyword arg parameters
    # @classmethod
    # def _get_param_names(cls):
    #     """Get parameter names for the estimator"""
    #     # fetch the constructor or the original constructor before
    #     # deprecation wrapping if any
    #     init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    #     if init is object.__init__:
    #         # No explicit constructor to introspect
    #         return []

    #     # introspect the constructor arguments to find the model parameters
    #     # to represent
    #     init_signature = inspect.signature(init)
    #     # Consider the constructor parameters excluding 'self'
    #     parameters = [p for p in init_signature.parameters.values()
    #                   if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    #     for p in parameters:
    #         if p.kind == p.VAR_POSITIONAL:
    #             raise RuntimeError("scikit-learn estimators should always "
    #                                "specify their parameters in the signature"
    #                                " of their __init__ (no varargs)."
    #                                " %s with constructor %s doesn't "
    #                                " follow this convention."
    #                                % (cls, init_signature))
    #     # Extract and sort argument names excluding 'self'
    #     return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


# TODO: probably rename this to something like TunableConfig
class Config(_Config):
    """
    Base class for tunable configs
    """
    def tune(self):
        """
        Returns the initialized tuning object for this config object.

        Output
        ------
        tuner: self or TunerConfig
            Returns a TuneConfig object if this config can be tuned. Otherwise resturns self for configs that cannot be tuned.
        """
        return self


class TunerConfig(Config):
    """
    Based class for a config tuner object.

    Parameters
    ----------
    base: Config
        The base config object to be tuned.
    """
    def __init__(self, base): pass

    def iter_configs(self, with_params=False):
        """
        Iterates over the tuning grid as a sequence of config objects.

        Parameters
        ----------
        with_params: bool
            Whether or not to include the unique tuning parameters for this tune setting.

        Yields
        ------
        config or (config, params) if with_params=True

        config: Config
            The config object for this tuning parameter setting.

        params: dict
            The unique tuning parameters for this setting.
        """
        config = deepcopy(self.base)
        for params in self.iter_params():
            config.set_params(**params)

            if with_params:
                yield config, params
            else:
                yield config

    def iter_params(self):
        """
        Iterates over the tuning grid as a sequence of dicts.

        Yields
        ------
        params: dict
            A dict containing the parameter values for this parameter setting.
        """
        raise NotImplementedError
        # for param in self.params:
        #     yield param


class TunerWithPathMixin:
    """
    Represents a tuned config object with a parameter path.

    Parameters
    ----------
    base: Config
        The base config object to be tuned.
    """

    # TODO: document this better -- also subclasses
    def iter_configs_with_path(self, with_params=False):
        """
        Iterates over the tuning parameter settings outputting the path parameters

        Parameters
        ----------
        with_params: bool
            Whether or not to include the unique tuning parameters for this tune setting.

        yields
        ------
        (config, path_lod) or (config, single_params, path_lod)

        config: Config
            The set config object with single parameters set.

        path_lod: iterable of dicts
            The list of dicts for the parameter path.

        single_params: dict
            The single parameter settings.
        """
        config = deepcopy(self.base)

        for single_param_settings, path_lod in self._iter_params_with_path():
            config.set_params(**single_param_settings)

            if with_params:
                yield config, single_param_settings, path_lod
            else:
                yield config, path_lod

    def iter_params(self):
        """
        Iterates over the tuning grid as a sequence of dicts.

        Yields
        ------
        params: dict
            A dict containing the parameter values for this parameter setting.
        """
        for to_yield, path_lod in self._iter_params_with_path():
            for path_params in path_lod:
                # to_yield.update(**path_params)
                yield {**to_yield, **path_params}

    def _iter_params_with_path(self):
        """
        Iterates over the tuning parameter settings outputting the path parameters

        yields
        ------
        single_param_settings, path_lod

        single_param_settings: dict
            The single parameter settings.

        path_lod: iterable of dicts
            The list of dicts for the parameter path.
        """
        raise NotImplementedError("Subclass should overwrite")


class ParamGridTuner(TunerConfig):
    """
    Tuner for a config with a parameter grid.
    """
    @autoassign
    def __init__(self, base, param_grid): pass

    def iter_params(self):
        for params in ParameterGrid(self.param_grid):
            yield params


class ManualTunerMixin:
    """
    Mixing for configs where we manually specify their tuning parameter grids via .tune().

    Class attributes
    ----------------
    _tunable_params: list of str
        The list of tunable parameters.
    """

    def tune(self, **params):
        """
        Sets the values for the parameters to be tuned.

        Parameters
        ----------
        **params:
            Each keyword argument input should be a list of parameter values. E.g. pen_val=np.arange(10). Only parameters listed under _tunable_params can be tuned.

        Output
        ------
        tuner: ParamGridTuner
            The tuner object.
        """

        # if we don't pass in any parameters just return self
        if len(params) == 0:
            return self

        #############################################
        # check we only tune the tunable parameters #
        #############################################

        # get names of all input parameters
        #if isinstance(param_grid, dict):
        input_params = list(params.keys())
        # else:
        #     input_params = set()
        #     for d in param_grid:
        #         input_params = input_params.union(d.keys())

        tunable_params = set(self._tunable_params)
        for name in input_params:
            if name not in tunable_params:
                raise ValueError("{} cannot be tuned. The tunable "
                                 "parameters are {}.".
                                 format(name, list(tunable_params)))

        return ParamGridTuner(base=self, param_grid=params)


def safe_get_config(config):
    """
    Safely returns the base config object when config maybe be either a config or a TunerConfig.
    """
    if isinstance(config, TunerConfig):
        return config.base
    else:
        return config
