from ya_glm.pen_max.lasso import get_pen_max as get_pen_max_lasso
from ya_glm.PenaltyConfig import get_convex_base_from_concave


def get_pen_max(X, y, loss,
                penalty, fit_intercept, sample_weight=None):
    """
    This ensures that 0 is a stationary point of the non-convex penalize problem. Of course no particualr algorithm is guarnteed to return 0 at this value due to the non-convexity.

    """

    if penalty.pen_func not in ['scad', 'mcp']:
        raise NotImplementedError("This heuristic only works for "
                                  "concave functions that look like a"
                                  "lasso a the origin")

    convex_pen = get_convex_base_from_concave(penalty)

    return get_pen_max_lasso(X=X, y=y, loss=loss,
                             penalty=convex_pen,
                             fit_intercept=fit_intercept,
                             sample_weight=sample_weight)
