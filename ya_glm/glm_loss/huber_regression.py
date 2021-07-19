from textwrap import dedent

from ya_glm.glm_loss.linear_regression import LinRegMixin
from ya_glm.glm_loss.linear_regression_multi_resp import \
    LinRegMultiRespMixin
from ya_glm.autoassign import autoassign


_huber_reg_params = dedent("""
knot: float
    The huber function knot parameter.
""")


class HuberRegMixin(LinRegMixin):
    is_multi_resp = False

    _loss_descr = dedent("""
    Huber regression with the huber loss L(z, y) = huber(z - y; knot) where

    huber(r; knot) = 0.5 r^2 if r <= knot
    huber(r; knot) = knot (abs(r) - 0.5 knot) if r > knot
    """)

    _params_descr = _huber_reg_params

    @autoassign
    def __init__(self, knot=1.35): pass

    def get_loss_info(self):
        loss_type = 'huber_reg'
        loss_kws = {'knot': self.knot}

        return loss_type, loss_kws


class HuberRegMultiRespMixin(LinRegMultiRespMixin):
    is_multi_resp = True

    _loss_descr = dedent("""
    Multiple response huber regression with the huber loss L(z, y) = sum_{j=1}^{n_responses} (z_j - y_j; knot) where

    huber(r; knot) = 0.5 r^2 if r <= knot
    huber(r; knot) = knot (abs(r) - 0.5 knot) if r > knot
    """)

    _params_descr = _huber_reg_params

    @autoassign
    def __init__(self, knot=1.35): pass

    def get_loss_info(self):
        loss_type = 'huber_reg_mr'
        loss_kws = {'knot': self.knot}

        return loss_type, loss_kws
