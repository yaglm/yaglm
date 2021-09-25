from copy import deepcopy
from ya_glm.config.base_params import ParamConfig, TunerConfig, \
     TunerWithPathMixin, get_base_config

from ya_glm.pen_max.non_convex import adjust_pen_max_for_non_convex
from ya_glm.pen_seq import get_pen_val_seq
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

    # TODO: perhaps remove this method since this information can be
    # accessed implicitly through the ya_glm.opt module. We will need
    # to modify ya_glm.opt a bit to make this in fact true.
    def get_func_info(self):
        """
        Returns information for this function e.g. is it proximable, smooth, etc

        Output
        ------
        info: dict
            Returns a dictionary with keys

            'smooth': bool
                Whether or not this penalty function is smooth.

            'proximable': bool
                Whether or not this has an easy to evaluate proximal opterator.

            'lin_proximable': bool
                Whether or not this penalty is a linear transformation away from being proximable e.g. the genearlized Lasso.
        """
        raise NotImplementedError("Subclass should overwrite")


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

        if hasattr(self, 'pen_val_max_'):
            max_val = self.pen_val_max_ * self.pen_max_mult
        else:
            max_val = None

        # get or standardize the penalty sequence.
        return get_pen_val_seq(n_pen_vals=self.n_pen_vals,
                               pen_vals=self.pen_val_seq,
                               pen_val_max=max_val,
                               pen_min_mult=self.pen_min_mult,
                               pen_spacing=self.pen_spacing)

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


###############
# Other utils #
###############

def get_unflavored(config):
    """
    Returns an unflavored version of the penalty config object. If the config is not flavored just retursn the original config; otherwise returns a copy.

    Parameters
    ----------
    config: PenaltyConfig
        The possibly flavored penalty config.

    Output
    ------
    config: PenaltyConfig
        Either the original penalty or an unflavored copy of the penalty.

    """
    if get_flavor_info(config) is not None:
        unflavored = deepcopy(config)
        unflavored.flavor = None
        return unflavored
    else:
        return config


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

    if not isinstance(base_config, WithFlavorPenSeqConfig) \
            or base_config.flavor is None:
        return None
    else:
        return get_base_config(base_config.flavor).name
