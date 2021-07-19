from textwrap import dedent

from ya_glm.base.Glm import Glm
from ya_glm.base.LossMixin import LossMixin


class GlmVanilla(LossMixin, Glm):

    _pen_descr = dedent("""
    Unpenalized GLM.
    """)

    pass
