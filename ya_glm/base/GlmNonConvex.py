from ya_glm.base.Glm import Glm
from ya_glm.pen_max.non_convex import get_pen_max


class GlmNonConvexPen(Glm):
    """
    Base class for GLMs with non-convex penalties.
    """

    def get_pen_val_max(self, X, y, sample_weight=None):
        """
        Returns the largest reasonable penalty parameter for the processed data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training covariate data.

        y: array-like, shape (n_samples, )
            The training response data.

        sample_weight: None or array-like,  shape (n_samples,)
            Individual weights for each sample.

        Output
        ------
        pen_val_max: float
            Largest reasonable tuning parameter value.
        """
        X_pro, y_pro, sample_weight_pro, _ = \
            self.preprocess(X, y, sample_weight=sample_weight, copy=True)

        return get_pen_max(X=X_pro, y=y_pro,
                           fit_intercept=self.fit_intercept,
                           sample_weight=sample_weight_pro,
                           loss=self._get_loss_config(),
                           penalty=self._get_penalty_config()
                           )
