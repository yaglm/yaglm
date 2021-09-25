from copy import deepcopy
from sklearn.model_selection import ParameterGrid

from ya_glm.config.base import Config
from ya_glm.autoassign import autoassign


class ParamConfig(Config):
    """
    Base class for tunable parameter configs.
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
    Base class for a config tuner object.

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
        config = tuners_to_base(config)  # flatten any tuner parameters

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

    def tune(self):
        """
        Just return self if .tune() gets called
        """
        return self


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
        config = tuners_to_base(config)  # flatten any tuner parameters

        for sps, path_lod in self._iter_params_with_path():
            config.set_params(**sps)

            if with_params:
                yield config, sps, path_lod
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
        for sps, path_lod in self._iter_params_with_path():
            for path_params in path_lod:
                yield {**sps, **path_params}

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


def get_base_config(config):
    """
    Safely returns the base config object when config maybe be either a config or a TunerConfig.

    Parameters
    ----------
    config: ParamConfig or TunerConfig
        An object that is either a ParamConfig or a TunerConfig.

    Output
    ------
    config: ParamConfig
        Either the input ParamConfig or the base ParamConfig from a TunerConfig.
    """
    if isinstance(config, TunerConfig):
        return config.base
    else:
        return config


def tuners_to_base(config):
    """
    For a config that is a TunerConfig or containts parameters that may be TunerConfigs
    this function replaces all the TunerConfigs with their base ParamConfigs.

    Note may modify the config object in place.

    Parameters
    ------
    config: ParamConfig, TunerConfig
        The config we want to modify.

    Output
    ------
    config: ParamConfig
        The modified config.
    """
    if isinstance(config, TunerConfig):
        return tuners_to_base(config.base)

    elif isinstance(config, ParamConfig):

        # replace any TunerConfig params
        for (k, v) in config.get_params(deep=False).items():
            config.set_params(**{k: tuners_to_base(v)})

        return config

    else:
        return config
