import numpy as np

from sklearn.utils import check_random_state
from scipy.sparse import random as random_sparse


def X_data_generator(random_state=0, hd=True, sparse=True,
                     n_samples=10, n_features=5):
    """
    Iterator yielding a sequence of X matrices to test.

    Parameters
    ----------
    random_state: int
        The seed.

    hd: bool
        Whether or not to yield the case n_features > n_samples

    sparse: bool
        Whether or not to yield a sparse matrix.

    """
    rng = check_random_state(random_state)

    # basic test
    yield rng.normal(size=(n_samples, n_features))

    # make sure one feature is no problem
    yield rng.normal(size=(10, 1))

    # n_features > n_samples works
    if hd:
        yield rng.normal(size=(n_samples, 2 * n_samples))

    if sparse:
        yield random_sparse(m=n_samples, n=n_features,
                            density=.5, random_state=rng).tocsr()


def lin_reg_y_from_X(X, n_responses=1, random_state=0):
    """
    Given a data matrix returns a matching response for linear regression.
    """
    rng = check_random_state(random_state)
    
    if n_responses == 1:
        return rng.normal(size=X.shape[0])
    
    else:
        return rng.normal(size=(X.shape[0], n_responses))


def log_reg_y_from_X(X, random_state=0):
    """
    Given a data matrix returns a matching response for logistic regression.
    """
    rng = check_random_state(random_state)
    return rng.choice(a=[0, 1], size=X.shape[0], p=[0.5, 0.5])


def max_norm(x):
    return abs(np.array(x).reshape(-1)).max()


def compare_fits(est, gt, tol=1e-2, behavior='error'):
    """
    Checks if two estimators have the same intercept and coefficient.
    """
    assert behavior in ['print', 'error']
    
    coef_diff = est.coef_ - gt.coef_
    if hasattr(est, 'intercept_'):
        inter_diff = est.intercept_ - gt.intercept_
    else:
        inter_diff = None

    coef_diff_norm = max_norm(coef_diff)
    if inter_diff is not None:
        inter_diff_norm = max_norm(inter_diff)

    if behavior == 'error':
        assert coef_diff_norm < tol, \
                'coef diff {}'.format(coef_diff_norm)

        if inter_diff is not None:
            assert inter_diff_norm < tol, \
                'intercept diff {}'.format(inter_diff_norm)

    elif behavior == 'print':
        if coef_diff_norm > tol:
            print('coef diff {}'.format(coef_diff_norm))

        if inter_diff is not None and inter_diff_norm > tol:
            print('intercept diff {}'.format(inter_diff_norm))
