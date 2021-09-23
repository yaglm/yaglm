from copy import deepcopy

from ya_glm.config.base import Config, TunerConfig, TunerWithPathMixin
from ya_glm.autoassign import autoassign

from ya_glm.pen_seq import get_pen_val_seq
from ya_glm.opt.penalty.nonconvex import get_nonconvex_func


class PenaltyConfig(Config):
    """
    Base GLM penalty config object.
    """

    @property
    def is_proximable(self):
        """
        Whether or not this penalty has an easy to evaluate proximal opterator. This helps us decide which algorithm to use by default
        """
        return True  # true by default because most penalties do!

    @property
    def is_smooth(self):
        """
        Whether or not this penalty is smooth. This helps us decide which algorithm to use by default.
        """
        return False  # false by default because most penalties are not!

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None):
        """
        Sets the tuning parameter sequence for a given dataset.
        """
        raise NotImplementedError

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
        raise NotImplementedError


class FlavoredMixin:
    """
    Mixin for flavored penalties.

    Parameters
    ----------
    flavor: None, FlavorConfig
    """

    def get_base_convex(self):
        """
        Gets the base convex penalty for the non-convex penalized problem, enforcing pen_val=1.

        Output
        ------
        cvx_pen: PenaltyConfig
            The base convex penalty
        """
        cvx_pen = deepcopy(self)
        cvx_pen.pen_val = 1.0
        cvx_pen.flavor = None
        return cvx_pen

    def get_non_smooth_transforms(self):
        """
        Returns the transformations for the non-smooth penalty terms e.g. the singular values for the nuclear norm.

        Output
        ------
        transf: None, callable(coef), dict of callables
            The transformation functions. Returns None if there are no non-smooth transforms. For a single penalty this returns a function that takes the coefficeint as input and spits out the non-smooth transformations. If there are multiple penalties this returns a dict of such callables.
        """
        raise NotImplementedError

    def get_transf_nonconex_penalty(self):
        """
        Returns the non-convex function that is applied to the transformation.
        """
        # TODO: something like this
        # assert self.flavor.is_nonconvex
        return get_nonconvex_func(name=self.flavor.pen_func,
                                  pen_val=self.pen_val,
                                  second_param=self.flavor.second_param_val)

    def get_pen_val_max(self, X, y, loss, fit_intercept=True,
                        sample_weight=None):

        pen_max = self._get_vanilla_pen_val_max(X=X, y=y, loss=loss,
                                                fit_intercept=fit_intercept,
                                                sample_weight=sample_weight)

        # possibly adjust pen val max for penalty flavoring
        if self.flavor is not None:
            pen_max = self.flavor.transform_pen_max(cvx_max_val=pen_max,
                                                    penalty=self)

        return pen_max

    def _get_vanilla_pen_val_max(self, X, y, loss, fit_intercept=True,
                                 sample_weight=None):
        raise NotImplementedError("Subclass should overwrite")


class PenaltyTuner(TunerConfig):
    """
    Represents a penalty parameter sequence/grid to be tuned over.

    Parameters
    ----------
    base: PenaltyConfig
        The base penalty config whose parameter(s) are to be tuned.
    """
    @autoassign
    def __init__(self, base): pass

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None):
        """
        Sets the tuning parameter sequence/grid for a given dataset.
        """
        raise NotImplementedError


class PenaltySeqTuner(TunerWithPathMixin, PenaltyTuner):

    """
    Config for a single penalty parameter sequence.

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
    pen_val_seq_: array-like
        The sequence of penalty values in decreasing order.

    pen_val_max_: float
        The largest reasonable penalty value. This is not computed if the user provides the penalty sequence.
    """

    @autoassign
    def __init__(self, base,
                 n_pen_vals=100, pen_val_seq=None,
                 pen_min_mult=1e-3, pen_spacing='log'): pass

    def set_tuning_values(self,  X, y, loss, fit_intercept=True,
                          sample_weight=None):
        """
        Sets the tuning parameter sequence for a given dataset. The tuning sequence is a either a linearly or logarithmically spaced sequence in the interval [pen_val_max * self.pen_min_mult, pen_val_max]. Logarithmic spacing means the penalty values are more dense close to zero.

        Parameters
        ----------
        TODO
        """
        max_val = None
        if self.pen_val_seq is None:

            # Compute the largest reasonable tuning parameter value
            try:
                self.pen_val_max_ = \
                    self.base.get_pen_val_max(X=X, y=y, loss=loss,
                                              fit_intercept=fit_intercept,
                                              sample_weight=sample_weight)

            except NotImplementedError:
                raise NotImplementedError("Please provide your "
                                          "own pen_val_seq; {} cannot "
                                          "automatically compute the "
                                          "largest reasonable penalty value".
                                          format(self.base))

            max_val = self.pen_val_max_

        # get or standardize the penalty sequence.
        self.pen_val_seq_ = get_pen_val_seq(n_pen_vals=self.n_pen_vals,
                                            pen_vals=self.pen_val_seq,
                                            pen_val_max=max_val,
                                            pen_min_mult=self.pen_min_mult,
                                            pen_spacing=self.pen_spacing)

        return self

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
        lod = [{'pen_val': pen_val} for pen_val in self.pen_val_seq_]
        yield {}, lod


class PenSeqTunerMixin:

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
