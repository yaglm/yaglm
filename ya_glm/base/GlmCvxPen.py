from ya_glm.base.Glm import Glm
from ya_glm.pen_max.lasso import get_pen_max
from ya_glm.pen_max.ridge import get_ridge_pen_max


class GlmCvxPen(Glm):
    """
    Base class for GLMs with convex penalties.
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
        X_pro, y_pro, _ = self.preprocess(X, y,
                                          sample_weight=sample_weight,
                                          copy=True)

        if self._primary_penalty_type == 'lasso':

            return get_pen_max(X=X_pro, y=y_pro,
                               fit_intercept=self.fit_intercept,
                               sample_weight=sample_weight,
                               loss=self._get_loss_config(),
                               penalty=self._get_penalty_config()
                               )

        elif self._primary_penalty_type == 'ridge':
            return get_ridge_pen_max(X=X_pro, y=y_pro,
                                     fit_intercept=self.fit_intercept,
                                     sample_weight=sample_weight,
                                     loss=self._get_loss_config(),
                                     penalty=self._get_penalty_config()
                                     )

        else:
            raise ValueError("Bad self._primary_penalty_type: {}"
                             "".format(self._primary_penalty_type))

    ################################
    # subclasses need to implement #
    ################################

    @property
    def _primary_penalty_type(self):
        """
        Is this primarily a ridge or a lasso penalty?
        """
        # 'lasso' or 'ridge'
        raise NotImplementedError
