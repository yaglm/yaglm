from ya_glm.pen_max.lasso import get_pen_max as get_pen_max_lasso
from ya_glm.PenaltyConfig import get_convex_base_from_concave


def get_pen_max(X, y,
                loss, penalty,
                fit_intercept, sample_weight=None):

    if not hasattr(penalty, 'coef_init'):
        raise ValueError("Penalty config must have a .coef_init attribute")

    coef_init = penalty.coef_init
    cvx_penalty = get_convex_base_from_concave(penalty)
    lasso_max_val = get_pen_max_lasso(X=X, y=y, loss=loss,
                                      penalty=cvx_penalty,
                                      fit_intercept=fit_intercept,
                                      sample_weight=sample_weight)

    if penalty.pen_func == 'scad':
        # TODO: allow this to depend on the penalty function
        # pushing the largest init elemnt under the pen param
        # forces all LLA weights to be equal to the pen param
        init_max = abs(coef_init).max()

        return max(init_max, lasso_max_val)

    else:
        # TODO: add this
        raise NotImplementedError
