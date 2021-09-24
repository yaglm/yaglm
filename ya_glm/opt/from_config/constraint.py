from ya_glm.config.constraint import Positive as PositiveConfig

from ya_glm.opt.constraint.convex import Positive


def get_constraint_func(config):

    if isinstance(config, PositiveConfig):
        return Positive()

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "ya_glm.opt.constraint".
                                  format(config))
