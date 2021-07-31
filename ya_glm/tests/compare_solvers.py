from ya_glm.solver.AndersonCdSolver import AndersonCdSolver
from ya_glm.solver.CvxpySolver import CvxpySolver
from ya_glm.solver.FistaSolver import FistaSolver
from ya_glm.solver.QuantileLQProgSolver import QuantileLQProgSolver

from ya_glm.tests.utils import lin_reg_y_from_X, log_reg_y_from_X, max_norm
from scipy.optimize import lsq_linear


from sklearn.utils import check_random_state


def compare_solvers(loss, penalty, fit_intercept=True,
                    behavior='error', tol=1e-2, random_state=0):

    """
    Check to see if different solvers give the same soluion
    """
    assert behavior in ['error', 'print']
    
    rng = check_random_state(random_state)
    
    if loss.name == 'log_reg' and penalty.lasso_pen_val is None and  \
            penalty.ridge_pen_val is None:
        
        # this makes unpenalized logisic regression stable by making the classes non separable
        n_samples = 100
        n_features = 2
    else:
        n_samples = 10
        n_features = 5
    
    X = rng.normal(size=(n_samples, n_features))

    if penalty.tikhonov is not None:
        penalty.tikhonov = rng.normal(size=(3, n_features))

    base = 'cvxpy'  # solver to compare to
    if loss.name in ['lin_reg', 'huber', 'quantile']:
        y = lin_reg_y_from_X(X)

    elif loss.name == 'log_reg':
        y = log_reg_y_from_X(X)

    coefs = {}

    if loss.name == 'lin_reg':

        solvers = {'fista': FistaSolver(),
                   'cvxpy': CvxpySolver()}

        if penalty.tikhonov is None:
            solvers['andy'] = AndersonCdSolver()

    elif loss.name == 'log_reg':

        solvers = {'fista': FistaSolver(),
                   'cvxpy': CvxpySolver()}

    elif loss.name == 'quantile':
        solvers = {'quantLQP': QuantileLQProgSolver(),
                   'cvxpy': CvxpySolver()}

    # fit each solver
    for name, solver in solvers.items():
        coef, intercept, opt_data = solver.solve(X.copy(), y.copy(),
                                                 loss=loss,
                                                 penalty=penalty,
                                                 fit_intercept=fit_intercept)
        coefs[name] = coef

        # scipy ground truth
        if loss.name == 'lin_reg' and not fit_intercept \
                and penalty.lasso_pen_val is None and \
                penalty.ridge_pen_val is None:
            coefs['scipy'] = lsq_linear(X, y).x

            base = 'scipy'  # ok here we have ground truth

    for name, coef in coefs.items():
        if name == base:
            continue

        diff = max_norm(coefs[name] - coefs[base])
    
        if behavior == 'print':
            if diff > tol:
                print('{} vs {}: {}'.format(name, base, diff))
        if behavior == 'error':
            assert diff < tol, \
                'diff = {} for {}'.format(diff, name)
