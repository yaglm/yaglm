from ya_glm.tests.zhu_admm_utils import check_simple, check_glm_lasso_reg, \
    generate_glm_data


def test_admm():
    kws = {'rho': 1}

    loss = 'lin_reg'
    pen_val = .3
    X, y = generate_glm_data(loss=loss)
    for D_mat in ['diag', 'prop_id']:
        kws['D_mat'] = D_mat

        check_simple(**kws)

        check_glm_lasso_reg(X=X, y=y, loss=loss, pen_val=pen_val,
                            tol=1e-3, **kws)
