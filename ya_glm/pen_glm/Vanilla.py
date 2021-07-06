from ya_glm.Glm import Glm


class GlmVanilla(Glm):

    def _get_solve_glm_kws(self):
        return {'loss_func': self._model_type,
                'fit_intercept': self.fit_intercept,
                **self.opt_kws
                }
