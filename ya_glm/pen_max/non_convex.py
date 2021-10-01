
def adjust_pen_max_for_non_convex(cvx_max_val, penalty, init_data):
    """
    Parameters
    ----------
    cvx_max_val: float
        The largest reasonable penalty value for the convex problem.

    penalty: PenaltyConfig

    init_data: dict
        The initializeation data. If we are using the LLA algorithm, this dict should have a key 'lla'.

    Output
    -----
    pen_max: float
        The possibly adjusted largest reasonable penalty parameter.
    """

    # check if we are using the LLA algorithm
    lla = init_data['lla']

    if lla:  # non-convex LLA

        # TODO-THINK-THROUGH: this aweful local import
        # gets around a circular import issue e.g. the from_config modules
        # import from base_penalty, but base_penalty imports this module
        # We should probably re-organize the code to get rid of this issue
        from ya_glm.opt.from_config.penalty import get_outer_nonconvex_func
        from ya_glm.opt.from_config.transforms import get_non_smooth_transforms

        transf = get_non_smooth_transforms(penalty)
        pen_func = get_outer_nonconvex_func(penalty)
        coef_init = init_data['coef']

        if not hasattr(pen_func, 'fcp_data'):
            raise NotImplementedError("TODO Need to add FCP data for"
                                      "this penalty")
        a1 = pen_func.fcp_data['a1']
        b1 = pen_func.fcp_data['b1']

        return max(abs(transf(coef_init)).max() / b1, cvx_max_val / a1)

    else:  # non-convex direct

        if penalty.flavor.pen_func in ['mcp', 'scad']:
            return cvx_max_val
        else:
            raise NotImplementedError("TODO add")
