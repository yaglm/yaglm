import numpy as np
from copy import deepcopy
from itertools import product

from ya_glm.config.base_params import ParamConfig, TunerConfig, \
     TunerWithPathMixin, get_base_config

from ya_glm.pen_max.non_convex import adjust_pen_max_for_non_convex
from ya_glm.pen_seq import get_sequence_decr_max, get_mix_val_seq
from ya_glm.autoassign import autoassign


class PenaltyConfig(ParamConfig):
    """
    Base class for all penalty configs
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


class PenaltyTuner(TunerConfig):
    """
    Base class for all penalty tuners
    """
    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None, init_data=None):
        """
        Sets the data needed to create the tuning parameter sequence/grid for a given dataset.
        """
        raise NotImplementedError("Subclass should overwrite")


####################
# Single penalties #
####################


class WithPenSeqConfig(PenaltyConfig):
    """
    Mixin for penalties that have a penalty value sequence.

    Attributes
    ----------
    pen_val: float
        The penalty value.
    """

    def tune(self, n_pen_vals=100,
             pen_min_mult=1e-3, pen_max_mult=1, pen_spacing='log',
             pen_val_seq=None):
        """
        Returns the tuning object for this penalty.

        Parameters
        ----------
        n_pen_vals: int
            Number of penalty values to try for automatically generated tuning parameter sequence. The default sequence lives in [pen_min_mult * pen_max_val, pen_max_val * pen_max_mult] where pen_max_val is an automatically computed largest reasonable penalty values.

        pen_min_mult: float
            Determines the smallest penalty value to try.

        pen_max_mult: float
            (Optional) Inflates the estimated largest reasonable penalty parameter value but a multiplicative factor. This is useful e.g. when the default estimated largest reasonable penalty parameter value is only an approximation.

        pen_spacing: str
            How the penalty values are spaced. Must be one of ['log', 'lin']
            for logarithmic and linear spacing respectively.

        pen_val_seq: None, array-like
            (Optional) User provided penalty value sequence.

        Output
        ------
        tuner: PenaltyParamTuner
            The penalty parameter tuning object.
        """
        kws = locals()
        kws['base'] = kws.pop('self')  # rename self to base
        return PenaltySeqTuner(**kws)

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None):
        """
        Computes the largest reasonable value for the penalty value for a given data set and loss.

        Parameters
        ----------
        TODO:

        Output
        ------
        pen_val_max: float or dict of floats
            The largest reasonable value for the penalty value. Returns a dict for mulitple penalties.
        """
        raise NotImplementedError("Subclass should overwrite")


class WithFlavorPenSeqConfig(WithPenSeqConfig):
    """
    Mixin for a penalty with flavoring.

    Attributes
    ----------
    pen_val: float
        The penalty value.

    flavor: None, FlavorConfig
        (Optional) Penalty flavor config.
    """

    def get_base_convex(self):
        """
        Gets the base convex penalty for the non-convex penalized problem, enforcing pen_val=1.

        Output
        ------
        cvx_pen: PenaltyConfig
            The base convex penalty
        """
        # TODO: add code from previous version
        pass

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None, init_data=None):

        pen_max = self._get_vanilla_pen_val_max(X=X, y=y, loss=loss,
                                                fit_intercept=fit_intercept,
                                                sample_weight=sample_weight)

        flavor_type = get_flavor_info(self)
        if flavor_type == 'adaptive' and self.weights is None:
            raise RuntimeError("The adaptive weights must be set before"
                               "calling get_pen_val_max()")

        # possibly adjust pen val max for penalty flavoring
        if flavor_type == 'non_convex':
            pen_max = adjust_pen_max_for_non_convex(pen_max,
                                                    penalty=self,
                                                    init_data=init_data)

        return pen_max

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):
        raise NotImplementedError("Subclass should overwrite")


###############################
# Tuners for single penalties #
###############################


class PenaltySeqTuner(TunerWithPathMixin, PenaltyTuner):
    """
    Config for a single penalty with a penalty parameter, pen_val, whose tuning grid is sequence.

    Parameters
    ----------
    base: PenaltyConfig
        The base penalty config.

    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence. The default sequence lives in [pen_min_mult * pen_max_val, pen_max_val * pen_max_mult] where pen_max_val is an automatically computed largest reasonable penalty values.

    pen_min_mult: float
        Determines the smallest penalty value to try.

    pen_max_mult: float
        (Optional) Inflates the estimated largest reasonable penalty parameter value but a multiplicative factor. This is useful e.g. when the default estimated largest reasonable penalty parameter value is only an approximation.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    pen_val_seq: None, array-like
        (Optional) User provided penalty value sequence.

    Attributes
    ----------
    pen_val_max_: float
        The largest reasonable penalty value. This is not computed if the user provides the penalty sequence.
    """

    @autoassign
    def __init__(self, base,
                 n_pen_vals=100,
                 pen_min_mult=1e-3, pen_max_mult=1,
                 pen_spacing='log', pen_val_seq=None): pass

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None, init_data=None):
        """
        Computes the largest reasonable value for pen_val.
        """

        # if the user has not provided a penalty value sequence then
        # compute the largest reasonable penalty parameter value.
        if self.pen_val_seq is None:

            # Compute the largest reasonable tuning parameter value
            try:
                self.pen_val_max_ = \
                    self.base.get_pen_val_max(X=X, y=y, loss=loss,
                                              fit_intercept=fit_intercept,
                                              sample_weight=sample_weight,
                                              init_data=init_data)

            except NotImplementedError:
                raise NotImplementedError("Please provide your "
                                          "own pen_val_seq; {} cannot "
                                          "automatically compute the "
                                          "largest reasonable penalty value".
                                          format(self.base))

    def get_pen_val_seq(self):
        """
        Returns the penalty value sequence in decreasing order.

        Output
        ------
        pen_val_seq: array-like, shape (n_pen_vals, )
            The penalty value sequence in decreasing order.
        """

        if self.pen_val_seq is not None:
            # ensure decreasing
            return np.sort(self.pen_val_seq)[::-1]
        else:
            max_val = self.pen_val_max_ * self.pen_max_mult

            # create sequence
            return get_sequence_decr_max(num=self.n_pen_vals,
                                         max_val=max_val,
                                         min_val_mult=self.pen_min_mult,
                                         spacing=self.pen_spacing)

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
        lod = [{'pen_val': pen_val} for pen_val in self.get_pen_val_seq()]
        yield {}, lod


############################################
# Configs for multiple, additive penalties #
############################################

class _AdditivePenaltyConfig(PenaltyConfig):
    """
    Base class for multiple additive penalties.

    For iterating over tuning grids we iterate over the penalties in order in which they appear. For path algorithms, the last penalty with a TunerConfig is used for the Path.
    """
    def tune(self):

        # call tune for each penalty
        for name, penalty in self.get_penalties().items():

            # this should modify the penalty in place!
            self.set_params(**{name: penalty.tune()})
        return AdditivePenaltyTuner(base=self)

    def get_penalties(self):
        """
        Returns the individual penalty configs. Like get_params(), these should be the actual penalty config objects so we can modify values by setting them in place.

        Output
        ------
        penalties: dict of PenaltyConfigs
            The individual penalty configs.
        """
        raise NotImplementedError("Subclass should overwrite")


class OverlappingSumConfig(_AdditivePenaltyConfig):
    """
    Represents the sum of several penalties that are fully overlapping.

    If flavored penalties are provided the flavors should not be mixed.

    For iterating over tuning grids we iterate over the penalties in order in which they appear. For path algorithms, the last penalty with a TunerConfig is used for the Path. For example, OverlappingSumConfig(ridge=Ridge(), lasso=Lasso()) would use the Lasso for the path algorithm.

    Parameters
    ----------
    **penalties
        PenaltyConfig objects passed in as keyword arguments. The keys will show up as attributes; none of the keys should end in '_'.

    Attributes
    ----------
    Each key passed into __init__'s **penalties keyword aguments will show up as an attribute.
    """
    @autoassign
    def __init__(self, **penalties): pass

    def get_penalties(self):
        return self.get_params(deep=False)


class InifmalSumConfig(_AdditivePenaltyConfig):
    """
    Represents the infimal sum of several penalties.

    Here the optimzation algorithm problem looks like

    min_{coef_1, ..., coef_q} L(sum_j coef_j) + sum_j pen_j(coef_j)

    which is equivalent to an infimal sum penalty on coef.

    If flavored penalties are provided the flavors should not be mixed.

    For iterating over tuning grids we iterate over the penalties in order in which they appear. For path algorithms, the last penalty with a TunerConfig is used for the Path. For example, InifmalSumConfig(ridge=Ridge(), lasso=Lasso()) would use the Lasso for the path algorithm.

    Parameters
    ----------
    **penalties
        PenaltyConfig objects passed in as keyword arguments. The keys will show up as attributes; none of the keys should end in '_'.

    Attributes
    ----------
    Each key passed into __init__'s **penalties keyword aguments will show up as an attribute.
    """
    @autoassign
    def __init__(self, **penalties): pass

    def get_penalties(self):
        return self.get_params(deep=False)


class SeparableSumConfig(_AdditivePenaltyConfig):
    """
    Represents penalties that are separable sum of mutiple penalties.

    If flavored penalties are provided, the flavors should not be mixed.

    For iterating over tuning grids we iterate over the penalties in order in which they appear. For path algorithms, the last penalty with a TunerConfig is used for the Path. For example, InifmalSumConfig(ridge=Ridge(), lasso=Lasso()) would use the Lasso for the path algorithm.

    Parameters
    ----------
    groups: dict of lists
        The lists of variables that are passed to each penalty. The keys in this dict should match the keys in the **penalties keyword arguments.

    **penalties
        PenaltyConfig objects passed in as keyword arguments. The keys will show up as attributes; none of the keys should end in '_'.

    Attributes
    ----------
    Each key passed into __init__'s **penalties keyword aguments will show up as an attribute.

    **penalties
        PenaltyConfig objects passed in as keyword arguments.
    """
    @autoassign
    def __init__(self, groups, **penalties): pass

    def get_penalties(self):
        """
        Returns the configs for each penalty.

        Output
        ------
        penalties: dict
            A dict containing the individual penalty configs.
        """
        params = self.get_params(deep=False)
        params.pop('groups')
        return params

    def get_groups(self):
        """
        Returns the group indices.

        Output
        ------
        groups: dict
            A dict containing the group indices. The keys are ordered according to get_penalties().keys()
        """
        return {k: self.groups[k] for k in self.get_penalties().keys()}

################################
# Tuner for multiple penalties #
################################


class AdditivePenaltyTuner(TunerWithPathMixin, PenaltyTuner):
    """
    Tuner for additive penalties.
    """

    @autoassign
    def __init__(self, base): pass

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None, init_data=None):
        kws = locals()
        kws.pop('self')

        # we may modify this below for separable sum penalties
        X = kws.pop(X)

        # setup each tunable penalty
        for name, penalty in self.base.get_penalties():

            # if a penalty is not a tuning object then skip it.
            if isinstance(penalty, PenaltyTuner):

                if isinstance(penalty.base, SeparableSumConfig):
                    # Tor separable sum penalties pretend that the X data is
                    # just the features corresponding to this group.
                    # This of course makes the pen_max computations
                    # only approximations!
                    feat_idxs = penalty.base.groups[name]
                    penalty.set_tuning_values(X=X[:, feat_idxs], **kws)

                else:
                    penalty.set_tuning_values(X=X, **kws)

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
        penalties = self.base.get_penalties()

        # we use the last tuned penalty for the path
        # all other penalties are treated as a grid
        path_pen_name = None
        for name, penalty in penalties.items():
            if isinstance(penalty, PenaltyTuner):
                path_pen_name = name

        # set path penalty, remove it from the penalties dict
        path_pen = penalties.pop(path_pen_name)

        # iterate over grid of each non-path penalty
        names = list(penalties.keys())
        for params in product(*list(pen.iter_params()
                              for pen in penalties.values())):
            # params is a tuple of dicts

            # convert to a single dict
            sps = {}  # single parameter settings
            for idx, name in enumerate(names):
                sps.update({name + '__' + k: v
                            for (k, v) in params[idx].items()})

            if path_pen_name is None:
                # of there is no path penalty then just yield the single
                # parameter settings
                yield sps, [{}]

            else:
                # iterate over path if there is path
                for path_sps, path_lod in path_pen._iter_params_with_path():

                    # update path single parameter setting names
                    path_sps = {path_pen_name + '__' + k: v
                                for (k, v) in path_sps.items()}

                    # update single parameter settings
                    sps = {**sps, **path_sps}

                    # update path lod names
                    new_path_lod = []
                    for path_params in path_lod:
                        new_path_params = {path_pen_name + '__' + k: v
                                           for (k, v) in path_params.items()}

                        new_path_lod.append(new_path_params)
                    yield sps, new_path_lod


##############
# ElasticNet #
##############


class ElasticNetConfig(PenaltyConfig):
    """
    Represents a penalty with an elastic net parameterization.

    first_{pen_val * mix_val}(coef) + second_{pen_val * (1 - mix_val)}(coef)

    Parameters
    ----------
    pen_val: float
        The penalty strength.

    mix_val: float
        The mixing parameter; must live in [0, 1].
    """

    @autoassign
    def __init__(self, pen_val=1, mix_val=0.5): pass

    def tune(self,
             n_pen_vals=100,
             pen_min_mult=1e-3,
             pen_max_mult=1,
             pen_spacing='log',
             pen_val_seq=None,

             n_mix_vals=10,
             mix_val_min=0.1,
             mix_val_seq=None):
        """
        Returns an ElasticNetTuner object. Note both the pen_val and mix_val are tuned over by default.

        Parameters
        ----------
        n_pen_vals: int
            Number of penalty values to try for automatically generated tuning parameter sequence. The default pen_val sequence lives in [pen_min_mult * pen_max_val, pen_max_val * pen_max_mult] where pen_max_val is an automatically computed largest reasonable penalty values.

        pen_min_mult: float
            Determines the smallest penalty value to try.

        pen_max_mult: float
            (Optional) Inflates the estimated largest reasonable penalty parameter value but a multiplicative factor. This is useful e.g. when the default estimated largest reasonable penalty parameter value is only an approximation.

        pen_spacing: str
            How the penalty values are spaced. Must be one of ['log', 'lin']
            for logarithmic and linear spacing respectively.

        pen_val_seq: None, array-like
            (Optional) User provided penalty value sequence.

        n_mix_vals: None, int
            Number of mix values to tune over in a sequence between [min_val_min, 1]. If None or 0 then mix_val (and mix_val_seq=None) the mix_val will not be tuned over.

        min_val_min: float
            The smallest mix_val value to tune over.

        mix_val_seq: None, array-like
            (Optional) User specified mix_val sequence to tune over.
        """
        kws = locals()
        kws['base'] = kws.pop('self')
        return ElasticNetTuner(**kws)

    def _get_sum_configs(self):
        """
        Gets the two penalty configs for the summed version e.g.

        Lasso(pen_val= pen_val * mix_val),
        Ridge(pen_val = pen_val * (1 - mix_val))

        Output
        ------
        first_config, second_config

        first_config: PenaltyConfig
            The penalty config for the first penalty.
        """
        raise NotImplementedError("Subclass should overwrite")


class ElasticNetTuner(TunerWithPathMixin, PenaltyTuner):
    """
    Tuner object for elastic net parameterized penalties.

    Parameters
    ----------
    base: ElasticNetConfig
        The base elastic net penalty config.

    n_pen_vals: int
        Number of penalty values to try for automatically generated tuning parameter sequence. The default pen_val sequence lives in [pen_min_mult * pen_max_val, pen_max_val * pen_max_mult] where pen_max_val is an automatically computed largest reasonable penalty values.

    pen_min_mult: float
        Determines the smallest penalty value to try.

    pen_max_mult: float
        (Optional) Inflates the estimated largest reasonable penalty parameter value but a multiplicative factor. This is useful e.g. when the default estimated largest reasonable penalty parameter value is only an approximation.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    pen_val_seq: None, array-like
        (Optional) User provided penalty value sequence.

    n_mix_vals: None, int
        Number of mix values to tune over in a sequence between [min_val_min, 1]. If None or 0 then mix_val (and mix_val_seq=None) the mix_val will not be tuned over.

    min_val_min: float
        The smallest mix_val value to tune over.

    mix_val_seq: None, array-like
        (Optional) User specified mix_val sequence to tune over.
    """
    @autoassign
    def __init__(self, base,
                 n_pen_vals=100,
                 pen_min_mult=1e-3,
                 pen_max_mult=1,
                 pen_spacing='log',
                 pen_val_seq=None,

                 n_mix_vals=10,
                 mix_val_min=0.1,
                 mix_val_seq=None): pass

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None, init_data=None):
        """
        Computes the largest reasonable value for pen_val.
        """

        kws = locals()
        kws.pop('self')

        first_config, second_config = self.base._get_sum_configs()

        # compute the largest reaonable penalty value for the primary config
        self.pen_max_first_ = first_config.get_pen_val_max(**kws)

        # only compute the largest reasonable penatly value for the
        # second penalty if we actually need it
        has_mix_val_0 = False

        if not self.tune_mix_vals and self.base.mix_val == 0:
            # if this is implicitly the second penalty only
            has_mix_val_0 = True

        elif self.tune_mix_vals:
            # if we are tuning the mix val

            if self.mix_val_min == 0 or \
                    (self.mix_val_seq is not None and  \
                     min(self.mix_val_seq) == 0):
                has_mix_val_0 = True

        if has_mix_val_0:
            self.pen_max_second_ = second_config.get_pen_val_max(**kws)

    @property
    def tune_mix_vals(self):
        """
        Output
        ------
        tune_mix_vals: bool
            Whether or not we are tuning the mix_val
        """
        if self.mix_val_seq is not None:
            return True
        elif self.n_mix_vals is not None and self.n_mix_vals >= 1:
            return True
        else:
            return False

    def get_mix_val_seq(self):
        """
        Returns the mix_val sequence.

        Output
        ------
        mix_val_seq: None, array-like
            The mix val sequence.
        """

        if self.tune_mix_vals:
            # if we are tuning the mix val
            if self.mix_val_seq is not None:
                return np.sort(np.array(self.mix_val_seq))

            else:
                return get_mix_val_seq(num=self.n_mix_vals,
                                       min_val=self.mix_val_min,
                                       spacing='log',
                                       prefer_larger=True)
        else:
            return None

    def get_pen_val_seq(self, mix_val):
        """
        Returns the penalty value sequence in decreasing order for a given mix_val.

        Parameters
        ----------
        mix_val: float
            The mix_val.

        Output
        ------
        pen_val_seq: array-like, shape (n_pen_vals, )
            The penalty value sequence in decreasing order.
        """

        if self.pen_val_seq is not None:
            # ensure decreasing
            return np.sort(self.pen_val_seq)[::-1]

        else:

            # get the largest reasonable penalty parameter
            if mix_val == 0:
                pen_max_val = self.pen_max_second_
            else:
                # adjust pen_max_val based on mix_val
                pen_max_val = self.pen_max_first_ / mix_val

            # possibly inflate pen max val
            pen_max_val = pen_max_val * self.pen_max_mult

            # create sequence
            return get_sequence_decr_max(num=self.n_pen_vals,
                                         max_val=pen_max_val,
                                         min_val_mult=self.pen_min_mult,
                                         spacing=self.pen_spacing)

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

        mix_val_seq = self.get_mix_val_seq()

        if self.tune_mix_vals:  # tune over both mix vals and pen vals
            # outer loop over mix vale, path over pen_vals
            for mix_val in mix_val_seq:
                # set mix val
                single_param_settings = {'mix_val': mix_val}

                # get pen val seq for this mix val
                pen_val_seq = self.get_pen_val_seq(mix_val=mix_val)
                pen_val_lod = [{'pen_val': pen_val} for pen_val in pen_val_seq]

                yield single_param_settings, pen_val_lod

        else:  # only tune over pen_val
            pen_val_seq = self.get_pen_val_seq(mix_val=self.base.mix_val)

            pen_val_lod = [{'pen_val': pen_val} for pen_val in pen_val_seq]
            yield {}, pen_val_lod

###############
# Other utils #
###############


# TODO: make this work for multi penalties
def get_flavor_info(config):
    """
    Gets the penalty flavor

    Parameters
    ----------
    config: PenaltyConfig, TunerConfig
        The penalty config or a TunerConfig.

    Output
    ------
    flavor_type: None, str
        The flavor type e.g. None, 'adaptive', or 'non_convex'
    """
    base_config = get_base_config(config)

    # ElasticNet Case
    if isinstance(base_config, ElasticNetConfig):
        # get flavors of component models
        flavors = [get_flavor_info(config)
                   for config in base_config._get_sum_configs()]

        # pull out the designated flavor
        return _flavor_from_multiple(flavors)

    # Additive Penalty case
    elif isinstance(base_config, _AdditivePenaltyConfig):

        # get flavors of component models
        flavors = [get_flavor_info(config)
                   for config in base_config.get_penalties().values()]

        # pull out the designated flavor
        return _flavor_from_multiple(flavors)

    # Single penalty case
    else:

        if isinstance(base_config, WithFlavorPenSeqConfig):
            # a flavorable penalty
            if base_config.flavor is None:
                return None
            else:
                return get_base_config(base_config.flavor).name
        else:
            # a non-flavorable penalty
            return None


def _flavor_from_multiple(flavors):
    """
    Given a list of flavors returns the desired flavor. We cannot mix flavors that are not None.

    Parameters
    ----------
    flavors: list
        The input flavors.

    Output
    ------
    flavor: None, str
        The flavor.
    """
    # remove None from flavors
    flavors = set(flavors).difference([None])
    if len(flavors) == 0:  # no flavors
        return None
    elif len(flavors) == 1:
        return list(flavors)[0]
    else:
        raise RuntimeError("Cannont mix non-None flavors!")


# TODO: make this work for multi penalties
def get_unflavored(config):
    """
    Returns an unflavored version of the penalty config object. If the config is not flavored just retursn the original config; otherwise returns a copy.

    Parameters
    ----------
    config: PenaltyConfig
        The possibly flavored penalty config.

    Outputd
    ------
    config: PenaltyConfig
        Either the original penalty or an unflavored copy of the penalty.
    """
    if isinstance(config, (ElasticNetConfig, _AdditivePenaltyConfig)):
        raise NotImplementedError("TODO: add this!")

    if get_flavor_info(config) is not None:
        unflavored = deepcopy(config)

        unflavored.set_params(flavor=None)
        return unflavored
    else:
        return config
