from yaglm.config.constraint import Positive as PositiveConfig

from yaglm.opt.constraint.convex import Positive


def get_constraint_func(config):

    if isinstance(config, PositiveConfig):
        return Positive()

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "yaglm.opt.constraint".
                                  format(config))
