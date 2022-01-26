from copy import deepcopy
from itertools import product

from yaglm.config.base_penalty import PenaltyTuner
from yaglm.adaptive import set_adaptive_weights
from yaglm.config.penalty_utils import build_penalty_tree, extract_penalties,\
    extract_flavors, get_flavor_kind
from yaglm.config.flavor import FlavorConfig


class PenaltyPerLossFlavorTuner:
    """
    Represents the tuning grid for the combinations os loss, penalty, penalty flavor, and constraint. This objects both sets up the tuning grids from data and handles iterating over the configs.

    Note one penalty grid is created for each loss/flavor combination (but not constraints).

    Parameters
    ----------
    loss: Config, TunerConfig
        Specifies the loss configuration or tuning grid.

    penalty: None, Config, TunerConfig
        (Optional) Specifies the penalty/flavor configuration or tuning grid.

    constraint: None, Config, TunerConfig
        (Optional) Specifies the constraint flavor configuration or tuning grid.

    Attributes
    ----------
    penalties_per_lf_: list of TunerCongifs
        The penalty TunerCongif config for each loss/flavor setting.

    """
    def __init__(self, loss, penalty=None, constraint=None):

        # set loss tuning grid
        self.loss = loss.tune()

        # set constraint tuning grid
        if constraint is not None:
            self.constraint = constraint.tune()
        else:
            self.constraint = None

        # set flavor tuning grid, store penalty object
        if penalty is not None:
            self.penalty = penalty.tune()
            self.flavor_grid = FlavorGrid(self.penalty)

        else:
            self.penalty = None
            self.flavor_grid = None

    def set_tuning_values(self, **kws):

        # if there is no penalty then we dont have to do anything
        if self.penalty is None:
            return self

        # list of the penalty tuners for each loss + flavor combination.
        self.penalties_per_lf_ = []

        # go over every loss/flavor combination
        for loss_config in maybe_iter_configs(self.loss):
            for flavor_configs in self.flavor_grid.iter_configs():

                # create a new base penalty config
                pen = deepcopy(self.penalty)

                # set flavor for this penalty
                if flavor_configs is not None:
                    pen.set_params(**flavor_configs)

                # set adaptive weights
                if get_flavor_kind(pen) in ['adaptive', 'mixed']:
                    pen = set_adaptive_weights(penalty=pen,
                                               init_data=kws['init_data'])

                # setup tuning values if this is a PenaltyTuner
                if isinstance(pen, PenaltyTuner):
                    kws['loss'] = loss_config
                    pen.set_tuning_values(**kws)

                self.penalties_per_lf_.append(pen)

        return self

    def iter_params(self):
        """
        Iterates over the tuning grid as a sequence of dicts.

        Yields
        ------
        params: dict of dicts
            A dict containing the parameter values for this parameter setting.

            params['loss']: dict
                The loss function parameters set here.

            params['penalty']: dict
                The penalty function parameters set here.

            params['constraint']: dict
                The constraint  parameters set here.

            params['flavor']: dict
                The flavor parameters set here. The flavor parameter keys map onto the non-tuned version of the base PenaltyConfig.
        """

        # setup loss/flavor iter (loss in outer loop)
        iter_loss = maybe_iter_params(self.loss)
        if self.flavor_grid is not None:
            iter_flavor = self.flavor_grid.iter_params()
        else:
            iter_flavor = [{}]
        lf_iter = product(iter_loss, iter_flavor)

        for lf_idx, (loss, flavor) in enumerate(lf_iter):

            # setup constraint iter
            constr_iter = maybe_iter_params(self.constraint)

            # the penalty corresponding to this loss/flavor combinations
            pen_tuner = self._get_penalty_tuner(lf_idx)
            pen_iter = maybe_iter_params(pen_tuner)

            # iterate over constraints/penalties
            # constraints in outer loop
            for (constr, penalty) in product(constr_iter, pen_iter):
                to_yield = {'loss': loss,
                            'flavor': flavor,
                            'constraint': constr,
                            'penalty': penalty}

                # drop anything that is not not actually set
                for k in list(to_yield.keys()):
                    if len(to_yield[k]) == 0:
                        to_yield.pop(k)

                yield to_yield

    def iter_configs(self, with_params=False):
        """
        Iterates over the tuning grid as a sequence of config objects. Note the flavor config is stored in the penalty config and therefore is not included in the returned config.

        Parameters
        ----------
        with_params: bool
            Whether or not to include the unique tuning parameters for this tune setting.

        Yields
        ------
        config or (config, params) if with_params=True

        config: dict of Config
            The dict of config object for this tuning parameter setting.

            config['loss']: dict of LossConfig
                The loss function config.

            config['penalty']: dict of PenaltyConfig
                The penalty config. Note the flavor parameters have been set here!

            config['constraint']: dict of ConstraintConfig
                The constraint config.

        params: dict of dicts
            The unique tune parameter settings. Note this DOES contain the flavor parameters!

            params['loss']: dict
                The loss function parameters set here.

            params['penalty']: dict
                The penalty function parameters set here.

            params['constraint']: dict
                The constraint  parameters set here.

            params['flavor']: dict
                The flavor parameters set here. The flavor parameter keys map onto the non-tuned version of the base PenaltyConfig.

        """

        for (loss_config, loss_params, pen_tuner,
                constr_config, constr_params, flavor_params) \
                in self._start_iter_configs(with_params=with_params):

            # setup penalty iterator
            penalty_iter = maybe_iter_configs(pen_tuner,
                                              with_params=with_params)

            # iterate over penalties
            for pen_config in penalty_iter:

                # break apart configs/params, setup params
                if with_params:
                    pen_config, pen_params = pen_config
                    params = {'loss': loss_params,
                              'penalty': pen_params,
                              'constraint': constr_params,
                              'flavor': flavor_params
                              }

                # config dict to return
                configs = {'loss': loss_config,
                           'penalty': pen_config,
                           'constraint': constr_config}

                if with_params:
                    yield configs, params
                else:
                    yield configs

    def iter_configs_with_pen_path(self, with_params=False):
        """
        Iterates over the tuning parameter settings outputting the path parameters.  See documentation for iter_configs() for more details.

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

        single_param_settings: dict
            The single parameter settings.
        """

        for (loss_config, loss_params, pen_tuner,
                constr_config, constr_params, flavor_params) \
                in self._start_iter_configs(with_params=with_params):

            # setup penalty iter
            penalty_iter = \
                    maybe_iter_configs_with_path(pen_tuner,
                                                 with_params=with_params)

            # iterate over penalties
            for pen_path_info in penalty_iter:

                # break apart configs/params, setup params
                if with_params:
                    pen_config, pen_single_params, pen_path_lod =\
                        pen_path_info

                    params = {'loss': loss_params,
                              'penalty': pen_single_params,
                              'constraint': constr_params,
                              'flavor': flavor_params}

                else:
                    pen_config, pen_path_lod = pen_path_info

                # config dict to return
                configs = {'loss': loss_config,
                           'penalty': pen_config,
                           'constraint': constr_config}

                if with_params:
                    yield configs, params, pen_path_lod
                else:
                    yield configs, pen_path_lod

    def _get_penalty_tuner(self, lf_idx):
        """
        Returns the penalty tuner corresponding to a given loss/flavor setting index.

        Parameters
        ----------
        lf_idx: int
            The loss-flavor setting index.

        Output
        ------
        pen_tuner: None, Config, ConfigTuner
            The penalty tuner for this loss/flavor index.
        """

        if hasattr(self, 'penalties_per_lf_'):
            # if we tune over penalties
            return self.penalties_per_lf_[lf_idx]
        else:
            return self.penalty

    def _start_iter_configs(self, with_params=False):
        """
        Used by both iter_configs and iter_configs_with_pen_path.

        Yields
        ------
        loss_config, loss_params, pen_tuner, \
            constr_config, constr_params, flavor_params
        """

        # setup loss/flavor iter (loss in outer loop)
        iter_loss = maybe_iter_configs(self.loss, with_params=with_params)
        if self.flavor_grid is not None:
            iter_flavor = self.flavor_grid.iter_params()  # the params, not configs!
        else:
            iter_flavor = [{}]
        lf_iter = product(iter_loss, iter_flavor)

        # outer loop over loss/flavor settings
        for lf_idx, (loss_config, flavor_params) in enumerate(lf_iter):

            # the penalty corresponding to this loss/flavor combinations
            pen_tuner = self._get_penalty_tuner(lf_idx)

            # maybe split configs/params
            if with_params:
                loss_config, loss_params = loss_config
            else:
                loss_params = None

            # setup constraint iter
            constr_iter = maybe_iter_configs(self.constraint,
                                             with_params=with_params)
            # iterate over constraints
            for constr_config in constr_iter:

                # maybe split configs/params
                if with_params:
                    constr_config, constr_params = constr_config
                else:
                    constr_params = None

                yield loss_config, loss_params, \
                    pen_tuner, \
                    constr_config, constr_params,\
                    flavor_params


def is_tuner(x):
    """
    Whether or not x is a tuner object (or just a config object).
    """
    return x is not None and hasattr(x, 'iter_params')


def maybe_iter_params(x):
    """
    Like x.iter_params(), but returns [{}] if x is just a config object and not a Tuner object.
    """
    if is_tuner(x):
        return x.iter_params()
    else:
        return [{}]


def maybe_iter_configs(x, with_params=False):
    """
    Like x.iter_configs(), but returns [x] or [(x, {})] if x is just a config object and not a Tuner object.
    """
    if is_tuner(x):
        return x.iter_configs(with_params=with_params)
    else:
        if with_params:
            # TODO: do we need the defensive copy? I had a reason I
            # can't remember now... Either figure out why and document or
            # get rid of it if we dont need it. Same applies to copies below
            return [(deepcopy(x), {})]
        else:
            return [deepcopy(x)]


def maybe_iter_configs_with_path(x, with_params=False):
    """
    Like x.maybe_iter_configs_with_path(), but returns [(x, [{}])] or [(x, {}, [{}])] if x is just a config object and not a Tuner object.
    """

    if is_tuner(x):
        return x.iter_configs_with_path(with_params=with_params)
    else:
        if with_params:
            return [(deepcopy(x), {}, [{}])]
        else:
            return [(deepcopy(x), {})]


class FlavorGrid:
    """
    Iterates over the flavor grid for a possibly multi penalty.

    Parameters
    ----------
    penalty: PenaltyConfig, PenaltyTuner
        The penalty whose flavors we want to tune over.

    Attributes
    ----------
    flavor_tree: dict
        This stores all the flavor configs as well as the information needed to propertly iterate over them. See build_flavor_tree().
    """
    def __init__(self, penalty):
        penalty_tree = build_penalty_tree(penalty)

        # pull of the flavor configs and keys corresponding to the tuner version
        penalties, keys = extract_penalties(penalty_tree, drop_tuners=False)
        self.flavors, self.keys_with_tuners = extract_flavors(penalties, keys)

        # also save the non-tuner version of the keys
        penalties, keys = extract_penalties(penalty_tree, drop_tuners=True)
        _, self.keys_no_tuners = extract_flavors(penalties, keys)

    def iter_configs(self):
        """
        Iterates over the flavor configs in the flavor grid. Each config maps onto the PenaltyTuner version of the penalty config.

        Yields
        ------
        configs: dict of configs
            The set flavor configs.
        """

        # make sure each flavor config can act like a tuner
        flavors = [_safe_wrap_tuner(f) for f in self.flavors]

        # TODO-HACK: I wanted to used the generators properly e.g.
        # for configs in product(*[f.iter_configs() for f in flavors]):
        # but I can't get it working.
        # Figure out how to do this more gracefully!!
        all_configs = [[deepcopy(c) for c in f.iter_configs()] for f in flavors]

        # iterate over the flavor tuning parameter grid settings
        for configs in product(*all_configs):
            yield {self.keys_with_tuners[idx]: c
                   for (idx, c) in enumerate(configs)}

    def iter_params(self):
        """
        Iterates over the tuned flavor params in the flavor grid. Each params maps onto the PenaltyConfig version of the penalty config. Only flavor parameters that are actually tuned over are returned.

        Yields
        ------
        params: dict
            Each value is a dict containing the parameter setting.
        """

        # make sure each flavor config can act like a tuner
        flavors = [_safe_wrap_tuner(f) for f in self.flavors]

        # TODO-HACK: see above in iter_configs
        all_params = [[deepcopy(c) for c in f.iter_params()] for f in flavors]

        # iterate over the flavor tuning parameter grid settings
        # for params in product(*[f.iter_params() for f in flavors]):
        for params in product(*all_params):

            # yield {self.keys_no_tuners[idx]: P for (idx, P)
            #        in enumerate(params) if len(P) > 0}
            #
            # format params
            to_yield = {}
            for (idx, P) in enumerate(params):
                # each P is a dict -- flatten it!
                if len(P) > 0:
                    base_key = self.keys_no_tuners[idx]
                    for (k, v) in P.items():
                        new_key = base_key + '__' + k
                        to_yield[new_key] = v
            yield to_yield


def _safe_wrap_tuner(flav):
    """
    Safely wraps a single flavor config in a tuner object that can iterate the way we want.
    """
    if isinstance(flav, FlavorConfig):
        return _SingleConfigTuner(flav)
    else:
        return flav


class _SingleConfigTuner:
    """
    An object that acts like a TunerConfig i.e. has iter_params and iter_configs but represents a single config
    object that is not tuned.
    """
    def __init__(self, config):
        self.config = config

    def iter_params(self):
        yield {}

    def iter_configs(self):
        yield self.config
