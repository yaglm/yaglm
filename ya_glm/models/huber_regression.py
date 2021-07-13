from ya_glm.models.linear_regression import LinRegMixin
from ya_glm.models.linear_regression_multi_resp import LinRegMultiResponseMixin
from ya_glm.autoassign import autoassign


class HuberRegMixin(LinRegMixin):

    @autoassign
    def __init__(self, knot=1.35): pass

    def get_loss_info(self):
        loss_type = 'huber_reg'
        loss_kws = {'knot': self.knot}

        return loss_type, loss_kws


class HuberRegMultiResponseMixin(LinRegMultiResponseMixin):
    @autoassign
    def __init__(self, knot=1.35): pass

    def get_loss_info(self):
        loss_type = 'huber_reg_mr'
        loss_kws = {'knot': self.knot}

        return loss_type, loss_kws
