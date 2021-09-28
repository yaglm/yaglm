from copy import deepcopy
from itertools import product

from ya_glm.config.base_params import get_base_config
from ya_glm.config.base_penalty import get_flavor_info
from ya_glm.adaptive import set_adaptive_weights


class PenaltyPerLossFlavorTuner:
    """
    Represents the tuning grid for the combinations os loss, penalty, penalty flavor, and constraint. This objects both sets up the tuning grids from data and handles iterating over the congigs.

    Note one penalty grid is created for each loss/flavor combination (but not constraints).

    Each parameter must be either a single config or a TunerConfig specifying the tuning grid.


    Parameters
    ----------
    loss: Config, TunerConfig
        Specifies the loss configuration or tuning grid.

    penalty: None, Config, TunerConfig
        (Optional) Specifies the penalty configuration or tuning grid.

    flavor: None, Config, TunerConfig
        (Optional) Specifies the penalty flavor configuration or tuning grid.

    constraint: None, Config, TunerConfig
        (Optional) Specifies the constraint flavor configuration or tuning grid.

    Attributes
    ----------
    penalties_per_lf_: list of TunerCongifs
        The penalty TunerCongif config for each loss/flavor setting.
    """

    def __init__(self, loss, penalty=None, constraint=None):
        # setup the tuned version of each config
        self.loss = loss.tune()

        self.penalty = None
        self.flavor = None
        self.constraint = None

        # maybe set the pnealty/flavor
        if penalty is not None:
            self.penalty = penalty.tune()

            # pull out flavor
            base_pen = get_base_config(self.penalty)
            flavor_kind = get_flavor_info(base_pen)
            if flavor_kind is not None:
                self.flavor = base_pen.flavor.tune()

        # maybe set the constraint
        if constraint is not None:
            self.constraint = constraint.tume()

    def set_tuning_values(self, **penalty_kws):
        """
        Sets up all tuning parameter sequences. Each loss/flavor setting gets its own penalty sequence.

        Parameters
        ----------
        **penalty_kws:
            Keyword arguments to penalty_tuner.set_tuning_values()

        Output
        ------
        self
        """

        if is_tuner(self.penalty):
            # setup penalty configs for each loss/flavor setting
            self.penalties_per_lf_ = []

            # iterate over loss/flavor combinations
            for lf_configs in self._iter_loss_flavor_configs():
                pen_tuner = deepcopy(self.penalty)
                base_pen = pen_tuner.base

                # set flavor for this penalty
                if 'flavor' in lf_configs:
                    base_pen.set_params(flavor=lf_configs['flavor'])

                ########################
                # Set adaptive weights #
                ########################
                if get_flavor_info(base_pen) == 'adaptive':

                    init_data = penalty_kws.get('init_data')

                    # set the adaptive weights
                    base_pen = set_adaptive_weights(penalty=base_pen,
                                                    init_data=init_data)

                # update penalty tuner
                pen_tuner.set_params(base=base_pen)

                # setup tuning parameter sequence
                pen_tuner.set_tuning_values(loss=lf_configs['loss'],
                                            **penalty_kws)

                self.penalties_per_lf_.append(pen_tuner)

        return self

    def get_penalty_tuner(self, lf_idx):
        """
        Returns the penalty tuner corresponding to a given loss/flavor setting index.

        Parameters
        ----------
        lf_idx: int
            The loss-flavor setting index.

        Output
        ------
        pen_tuner: None, Config, ConfigTuner

        """

        if hasattr(self, 'penalties_per_lf_'):
            # if we tune over penalties
            return self.penalties_per_lf_[lf_idx]
        else:
            return self.penalty

    def iter_params(self):
        """
        Iterates over the tuning grid as a sequence of dicts.

        Yields
        ------
        params: dict
            A dict containing the parameter values for this parameter setting. The keys of this dict may include ['loss', 'flavor', 'penalty'] and the entries of this dict are dicts with the corresponding parameter setting.
        """

        # iterate over loss/flavor combinations
        for idx, lf_params in enumerate(self._iter_loss_flavor_params()):

            # the penalties correspond to loss/flavor combinations
            pen_tuner = self.get_penalty_tuner(idx)

            # setup constraint iter
            constr_iter = maybe_iter_params(self.constraint)

            # iterate over constraints
            for constr in constr_iter:

                # iterate over penalties
                for penalty in pen_tuner.iter_params():
                    yield {**lf_params,
                           'constraint': constr,
                           'penalty': penalty}

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
            The dict of config object for this tuning parameter setting. The enries of the dict may include ['loss', 'flavor', 'penalty'].

        params: dict
            The unique tune parameter settings.
        """

        loss_flavor_iter = \
            self._iter_loss_flavor_configs(with_params=with_params)

        # iterate over loss/flavor combinations
        for lf_idx, lf_configs in enumerate(loss_flavor_iter):

            # the penalties correspond to loss/flavor combinations
            pen_tuner = self.get_penalty_tuner(lf_idx)

            # maybe split configs/params
            if with_params:
                lf_configs, lf_params = lf_configs

            # drop the flavor config since it is included in the penalty config
            lf_configs.pop('flavor', None)

            # setup constraint iter
            constr_iter = maybe_iter_configs(self.constraint,
                                             with_params=with_params)

            # iterate over constraints
            for constr_config in constr_iter:

                # maybe split configs/params
                if with_params:
                    constr_config, constr_params = constr_config

                # setup penalty iterator
                penalty_iter = maybe_iter_configs(pen_tuner,
                                                  with_params=with_params)

                # iterate over penalties
                for pen_config in penalty_iter:

                    # break apart configs/params, setup params
                    if with_params:
                        pen_config, pen_params = pen_config
                        params = {**lf_params,
                                  'penalty': pen_params,
                                  'constraint': constr_params}

                    # config dict to return
                    configs = {**lf_configs,
                               'penalty': pen_config,
                               'constraint': constr_config}

                    if with_params:
                        yield configs, params
                    else:
                        yield configs

    def iter_configs_with_pen_path(self, with_params=False):
        """
        Iterates over the loss/flavor configs with penalty paths

        TODO: document this
        """
        loss_flavor_iter = \
            self._iter_loss_flavor_configs(with_params=with_params)

        # iterate over loss/penalty combinations #
        for lf_idx, lf_configs in enumerate(loss_flavor_iter):

            # the penalties correspond to loss/flavor combinations
            pen_tuner = self.get_penalty_tuner(lf_idx)

            # maybe split configs/params
            if with_params:
                lf_configs, lf_params = lf_configs

            # drop the flavor config since it is included in the penalty config
            lf_configs.pop('flavor', None)

            # setup constraint iter
            constr_iter = maybe_iter_configs(self.constraint,
                                             with_params=with_params)
            # iterate over constraints #
            for constr_config in constr_iter:

                # maybe split configs/params
                if with_params:
                    constr_config, constr_params = constr_config

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

                        params = {**lf_params,
                                  'penalty': pen_single_params,
                                  'constraint': constr_params}

                    else:
                        pen_config, pen_path_lod = pen_path_info

                    # config dict to return
                    configs = {**lf_configs,
                               'penalty': pen_config,
                               'constraint': constr_config}

                    if with_params:
                        yield configs, params, pen_path_lod
                    else:
                        yield configs, pen_path_lod

    def _iter_loss_flavor_params(self):
        """
        Iterates of the loss/flavor parameter settings.
        """
        # setup iterators
        loss_iter = maybe_iter_params(self.loss)
        flavor_iter = maybe_iter_params(self.flavor)

        # iterate over loss/flavor settings
        for loss, flavor in product(loss_iter, flavor_iter):
            yield {'loss': loss, 'flavor': flavor}

    def _iter_loss_flavor_configs(self, with_params=False):
        """
        Iterates of the loss/flavor configs.

        TODO: document
        """

        # setup iterators
        loss_iter = maybe_iter_configs(self.loss, with_params=with_params)
        flavor_iter = maybe_iter_configs(self.flavor, with_params=with_params)

        for loss, flavor in product(loss_iter, flavor_iter):

            if with_params:
                configs = {'loss': loss[0], 'flavor': flavor[0]}
                params = {'loss': loss[1], 'flavor': flavor[1]}
                yield configs, params
            else:

                configs = {'loss': loss, 'flavor': flavor}
                yield configs


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


def maybe_iter_configs(x, with_params):
    """
    Like x.iter_configs(), but returns [x] or [(x, {})] if x is just a config object and not a Tuner object.
    """
    if is_tuner(x):
        return x.iter_configs(with_params=with_params)
    else:
        if with_params:
            return [(deepcopy(x), {})]
        else:
            return [deepcopy(x)]


def maybe_iter_configs_with_path(x, with_params):
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
