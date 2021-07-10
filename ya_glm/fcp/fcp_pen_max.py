from ya_glm.glm_pen_max_lasso import lasso_max


def get_fcp_pen_val_max(X, y, init_data, loss_func, loss_kws,
                        pen_func, pen_func_kws,
                        fit_intercept=True):
    """
    Returns the largest reasonable tuning parameter value for fitting a
    folded concave penalty with the LLA algorithm. Larger penalty values
    will result in the LLA algorithm converging to zero.

    Parameters
    ----------
    X:

    y:

    loss_func: str

    loss_kws: dict

    init_data: dict

    pen_func: str

    pen_kws: dict

    fit_intercept: bool
    """

    lasso_max_val = lasso_max(X=X, y=y,
                              fit_intercept=fit_intercept,
                              loss_func=loss_func,
                              loss_kws=loss_kws)

    if pen_func == 'scad':
        # TODO: allow this to depend on the penalty function
        # pushing the largest init elemnt under the pen param
        # forces all LLA weights to be equal to the pen param
        init_max = abs(init_data['coef']).max()

        return max(init_max, lasso_max_val)

    else:
        # TODO: add this
        raise NotImplementedError
