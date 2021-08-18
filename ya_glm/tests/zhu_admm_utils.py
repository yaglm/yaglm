import numpy as np

from ya_glm.opt.zhu_admm import solve
from ya_glm.opt.fista import solve_fista
from ya_glm.admm.GlmInputLoss import LinRegInput
from ya_glm.opt.base import Zero
from ya_glm.opt.glm_loss.linear_regression import LinReg
from ya_glm.opt.penalty.vec import LassoPenalty
from ya_glm.toy_data import sample_sparse_lin_reg


def check_simple(tol=1e-2, **kws):
    y = np.ones(2)
    A1 = np.eye(2)
    A2 = np.zeros((2, 2))

    g1 = LinRegInput(y=y)
    g2 = Zero()

    soln_admm, opt_out, admm_out = solve(A1=A1, A2=A2, g1=g1, g2=g2, **kws)

    diff_norm = max(abs(soln_admm - y)).max()
    assert diff_norm < tol


def check_glm_lasso_reg(X, y, loss, pen_val, tol=1e-3, **kws):

    if loss == 'lin_reg':
        smooth_loss = LinReg(X=X, y=y, fit_intercept=False)
        init_val = np.zeros(X.shape[1])
        g1 = LinRegInput(y=y)

    else:
        raise NotImplementedError()

    L1 = LassoPenalty(mult=pen_val)

    fista_soln = solve_fista(smooth_func=smooth_loss,
                             non_smooth_func=L1,
                             init_val=init_val)[0]

    soln_admm, opt_out, admm_out = \
        solve(g1=g1, g2=L1, A1=X, A2=np.eye(X.shape[1]),
              **kws)

    diff_norm = max(abs(fista_soln - soln_admm))
    assert diff_norm <= tol


def generate_glm_data(loss='lin_reg'):
    """
    Returns
    -------
    X, y

    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.
    """
    n_samples = 100
    n_features = 10
    random_state = 0

    if loss == 'lin_reg':
        return sample_sparse_lin_reg(n_samples=n_samples, n_features=n_features,
                                     random_state=random_state)[0:2]

    else:
        raise NotImplementedError()
