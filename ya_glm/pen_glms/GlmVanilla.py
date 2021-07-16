from ya_glm.base.Glm import Glm


class GlmVanilla(Glm):

    def _get_solve_kws(self):
        loss_func, loss_kws = self.get_loss_info()

        return {'loss_func': loss_func,
                'loss_kws': loss_kws,

                'fit_intercept': self.fit_intercept,
                **self.opt_kws,
                }
