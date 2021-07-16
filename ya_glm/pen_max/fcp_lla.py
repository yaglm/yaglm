from ya_glm.pen_max.lasso import get_pen_max as get_pen_max_lasso


def get_pen_max(X, y, init_data, pen_func, pen_func_kws,
                loss_func, loss_kws={},
                groups=None,
                fit_intercept=True,
                pen_kind='entrywise'):
    """
    Returns the largest reasonable tuning parameter value for fitting a
    folded concave penalty with the LLA algorithm. Larger penalty values
    will result in the LLA algorithm converging to zero.

    Parameters
    ----------
    X

    y

    init_data

    pen_func

    pen_func_kws:

    loss_func:

    loss_kws:

    groups:

    fit_intercept:

    pen_kind:

    """

    kws = {'X': X,
           'y': y,
           'fit_intercept': fit_intercept,
           'loss_func': loss_func,
           'loss_kws': loss_kws}

    if pen_kind == 'group':
        kws['groups'] = groups

    lasso_max_val = get_pen_max_lasso(pen_kind=pen_kind, **kws)

    if pen_func == 'scad':
        # TODO: allow this to depend on the penalty function
        # pushing the largest init elemnt under the pen param
        # forces all LLA weights to be equal to the pen param
        init_max = abs(init_data['coef']).max()

        return max(init_max, lasso_max_val)

    else:
        # TODO: add this
        raise NotImplementedError
