
_MULTI_RESP_LOSSES = ['lin_reg_mr', 'huber_reg_mr',
                      'multinomial', 'poisson_mr', 'quantile_mr']

_MULTI_RESP_PENS = ['multi_task_lasso', 'multi_task_lasso_enet',
                    'nuclear_norm']


def is_multi_response(loss_func):
    return loss_func in _MULTI_RESP_LOSSES