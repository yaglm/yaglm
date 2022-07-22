from yaglm.config.constraint import Positive as PositiveConfig
from yaglm.config.constraint import Simplex as SimplexConfig
from yaglm.config.constraint import DevecPSD as DevecPSDConfig

from yaglm.opt.constraint.convex import Positive, Simplex
from yaglm.opt.constraint.psd import Devec2SymMat, PSDCone


def get_constraint_func(config):

    if isinstance(config, PositiveConfig):
        return Positive()

    elif isinstance(config, SimplexConfig):
        return Simplex(radius=config.radius)

    elif isinstance(config, DevecPSDConfig):
        return Devec2SymMat(d=config.d, func=PSDCone(force_sym=True))

    else:
        raise NotImplementedError("{} is not currently supported by "
                                  "yaglm.opt.constraint".
                                  format(config))
