from itertools import product


from ya_glm.loss.LossConfig import LinReg, LogReg, Quantile
from ya_glm.PenaltyConfig import ConvexPenalty
from ya_glm.tests.compare_solvers import compare_solvers


def test_solvers(tol=1e-2):
    losses = [LinReg(), LogReg(), Quantile(quantile=.2)]

    penalties = [ConvexPenalty(),
                 ConvexPenalty(lasso_pen_val=.1),
                 ConvexPenalty(lasso_pen_val=.1, ridge_pen_val=.2),
                 ConvexPenalty(ridge_pen_val=.2),

                 # Tikhonov is giving issues
                 # TODO: figure out whats going on
                 # ConvexPenalty(lasso_pen_val=.1, ridge_pen_val=.2,
                 #               tikhonov=True)  # this will be set in compare_solvers
                 ]

    for (loss, penalty, fit_intercept) in product(losses, penalties,
                                                  [True, False]):

        compare_solvers(loss=loss, penalty=penalty, tol=tol)
