from ya_glm.opt.penalty.vec import LassoPenalty, RidgePenalty,\
    LassoRidgePenalty

from ya_glm.tests.opt_checking_utils import check_nonsmooth, \
    check_smooth_with_prox, penalty_value_func_gen,\
    penalty_value_func_get_l1_l2


def test_lasso():
    for idx, (kws, values) in enumerate(penalty_value_func_gen()):
        check_nonsmooth(func=LassoPenalty(**kws),
                        values=values)


def test_ridge():
    for idx, (kws, values) in enumerate(penalty_value_func_gen()):
        check_smooth_with_prox(func=RidgePenalty(**kws),
                               values=values)


def test_lasso_ridge():
    for idx, (kws, values) in enumerate(penalty_value_func_get_l1_l2()):
        check_nonsmooth(func=LassoRidgePenalty(**kws),
                        values=values)
