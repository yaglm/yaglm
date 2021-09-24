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

    def tune(self, n_pen_vals=100, pen_val_seq=None,
             pen_min_mult=1e-3, pen_spacing='log'):
        """
        Returns the tuning object for this penalty.

        Parameters
        ----------
        n_pen_vals: int
            Number of penalty values to try for automatically generated tuning parameter sequence.

        pen_val_seq: None, array-like
            (Optional) User provided penalty value sequence.

        pen_min_mult: float
            Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

        pen_spacing: str
            How the penalty values are spaced. Must be one of ['log', 'lin']
            for logarithmic and linear spacing respectively.

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
        Number of penalty values to try for automatically generated tuning parameter sequence.

    pen_val_seq: None, array-like
        (Optional) User provided penalty value sequence.

    pen_min_mult: float
        Determines the smallest penalty value to try. The automatically generated penalty value squence lives in the interval [pen_min_mult * pen_max_val, pen_max_val] where pen_max_val is automatically determined.

    pen_spacing: str
        How the penalty values are spaced. Must be one of ['log', 'lin']
        for logarithmic and linear spacing respectively.

    Attributes
    ----------
    pen_val_max_: float
        The largest reasonable penalty value. This is not computed if the user provides the penalty sequence.
    """

    @autoassign
    def __init__(self, base,
                 n_pen_vals=100, pen_val_seq=None,
                 pen_min_mult=1e-3, pen_spacing='log'): pass

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

        if hasattr(self, 'pen_val_max_'):
            max_val = self.pen_val_max_
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


######################
# Multiple penalties #
######################

# # TODO: do we need this super class?
# class _AdditivePenaltyConfig(PenaltyConfig):
#     """
#     Base class for multiple added  penalties.

#     Attributes
#     ----------
#     penalties_
#     """
#     def tune(self):

#         # call tune for each penalty
#         for i, pen in enumerate(self.penalties_):
#             self.penalties_[i] = pen.tune()

#         return AdditivePenaltyTuner(base=self)

#     def get_base_configs(self):
#         """
#         Output
#         ------
#         penalties: list of PenaltyConfig's
#         """
#         raise NotImplementedError("Subclass should overwrite")


# class OverlappingSumConfig(_AdditivePenaltyConfig):
#     """
#     Represents the sum of several penalties that are fully overlapping.
#     """

#     def get_func_info(self):

#         infos = [pen.get_func_info() for pen in self.get_base_configs()]

#         # smooth if all smooth
#         smooth = all(info['smooth'] for info in infos)
#         prox = False  # False by default -- this may be true for some subclasses

#         # linear proximable if all
#         lin_prox = all((info['proximable'] or info['lin_proximable'])
#                        for info in infos)

#         return {'smooth': smooth,
#                 'proximable': prox,
#                 'lin_proximable': lin_prox}


# class InifmalSumConfig(_AdditivePenaltyConfig):
#     """
#     Represents the infimal sum of several penalties.
#     """

#     def get_func_info(self):

#         infos = [pen.get_func_info() for pen in self.get_base_configs()]

#         # smooth if all smooth
#         smooth = all(info['smooth'] for info in infos)

#         # proximable if all are proximable
#         prox = all(info['proximable'] for info in infos)

#         # linear proximable if all are lin proximable
#         lin_prox = all((info['proximable'] or info['lin_proximable'])
#                        for info in infos)

#         return {'smooth': smooth,
#                 'proximable': prox,
#                 'lin_proximable': lin_prox}


# class SeparableSumConfig(_AdditivePenaltyConfig):
#     """
#     Represents penalties that are separable sum of mutiple penalties.
#     """
#     def get_func_info(self):

#         infos = [pen.get_func_info() for pen in self.get_base_configs()]

#         # smooth if all smooth
#         smooth = all(info['smooth'] for info in infos)

#         # proximable if all are proximable
#         prox = all(info['proximable'] for info in infos)

#         # linear proximable if all are lin proximable
#         lin_prox = all((info['proximable'] or info['lin_proximable'])
#                        for info in infos)

#         return {'smooth': smooth,
#                 'proximable': prox,
#                 'lin_proximable': lin_prox}


# ################################
# # Tuner for multiple penalties #
# ################################

# class AdditivePenaltyTuner(TunerConfig):
#     """
#     Tuner for additive penalties.
#     """

#     @autoassign
#     def __init__(self, base): pass

#     def set_tuning_values(self,  X, y, loss, fit_intercept=True,
#                           sample_weight=None, init_data=None):
#         kws = locals()
#         kws.pop('self')

#         # setup each tunable penalty
#         for i, pen in enumerate(self.base.penalties_):

#             # if a penalty is not a tuning object then skip it.
#             if isinstance(pen, PenaltyTuner):
#                 pen.set_tuning_values(**kws)
#                 self.base.penalties_[i] = pen

# ##############
# # ElasticNet #
# ##############


# class ElasticNetConfig(PenaltyConfig):
#     """
#     Represents a penalty with an elastic net parameterization.
#     """
#     pass


# class ElasticNetTuner(TunerConfig):
#     """
#     Tuner object for elastic net parameterized penalties.
#     """
#     def set_tuning_values(self,  X, y, loss, fit_intercept=True,
#                           sample_weight=None, init_data=None):
#         raise NotImplementedError("TODO: add")


###############
# Other utils #
###############

def get_unflavored(config):
    if get_flavor_info(config) is not None:
        unflavored = deepcopy(config)
        unflavored.flavor = None
        return unflavored
    else:
        return config


def get_flavor_info(config):
    """
    Output
    ------
    flavor_type
    """
    base_config = get_base_config(config)

    if not isinstance(base_config, WithFlavorPenSeqConfig) \
            or base_config.flavor is None:
        return None
    else:
        return get_base_config(base_config.flavor).name
