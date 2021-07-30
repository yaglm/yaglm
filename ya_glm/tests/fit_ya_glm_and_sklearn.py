from ya_glm.models.Vanilla import Vanilla
from ya_glm.models.Lasso import Lasso
from ya_glm.models.Ridge import Ridge
from ya_glm.models.ENet import ENet
from ya_glm.loss.LossConfig import LogReg

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso as LassoSK
from sklearn.linear_model import Ridge as RidgeSK
from sklearn.linear_model import ElasticNet as ENetSk

from sklearn.linear_model import LogisticRegression


def fit_lin_reg(X, y,
                fit_intercept=True, standardize=False,
                sample_weight=None,
                lasso_pen_val=None,
                ridge_pen_val=None,
                solver='default',
                ya_kws={},
                sk_kws={}):
    """
    Fits a ya_glm and sklearn linear regression estimator with the same arguments to a given dataset.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    standardize: bool
        Whether or not to perform internal standardization before fitting the data.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    lasso_pen_val: None, float
        The Lasso penalty value, if desired.

    ridge_pen_val: None, float
        The Lasso penalty value, if desired.

    solver:
        The solver for ya_glm

    ya_kws: dict
        Additional keyword arguments for the ya_glm models.

    sk_kws: dict
        Additional keyword arguments for the sklearn models.

    Output
    ------
    est, sk

    est:
        The fit ya_glm estimator

    sk:
        The fit sklearn estimator
    """

    n_samples = X.shape[0]
    if lasso_pen_val is None and ridge_pen_val is None:
        ######################
        # Non-penalized case #
        ######################

        est = Vanilla(fit_intercept=fit_intercept,
                      standardize=standardize,
                      solver=solver,
                      **ya_kws)

        sk = LinearRegression(fit_intercept=fit_intercept,
                              normalize=standardize,
                              **sk_kws)

    elif lasso_pen_val is not None and ridge_pen_val is not None:

        ###############
        # Elastic net #
        ###############

        pen_val = lasso_pen_val + ridge_pen_val
        l1_ratio = lasso_pen_val / pen_val

        est = ENet(pen_val=pen_val,
                   l1_ratio=l1_ratio,
                   fit_intercept=fit_intercept,
                   standardize=standardize,
                   solver=solver,
                   **ya_kws)

        sk = ENetSk(alpha=pen_val,
                    l1_ratio=l1_ratio,
                    fit_intercept=fit_intercept,
                    normalize=standardize,
                    **sk_kws)

    elif lasso_pen_val is not None:
        #########
        # Lasso #
        #########

        est = Lasso(pen_val=lasso_pen_val,
                    fit_intercept=fit_intercept,
                    standardize=standardize,
                    solver=solver,
                    **ya_kws)

        sk = LassoSK(alpha=lasso_pen_val,
                     fit_intercept=fit_intercept,
                     normalize=standardize,
                     **sk_kws)

    else:
    
        ##########
        # Ridge #
        #########

        est = Ridge(pen_val=ridge_pen_val,
                    fit_intercept=fit_intercept,
                    standardize=standardize,
                    solver=solver,
                    **ya_kws)

        sk = RidgeSK(alpha=ridge_pen_val * n_samples,
                     fit_intercept=fit_intercept,
                     normalize=standardize,
                     **sk_kws)

    est.fit(X, y, sample_weight=sample_weight)
    sk.fit(X, y, sample_weight=sample_weight)
    
    return est, sk
   

def fit_log_reg(X, y, fit_intercept=True,
                sample_weight=None,
                lasso_pen_val=None,
                ridge_pen_val=None,
                class_weight=None,
                solver='default',
                ya_kws={}, sk_kws={}):

    """
    Fits a ya_glm and sklearn linear regression estimator with the same arguments to a given dataset.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The training covariate data.

    y: array-like, shape (n_samples, ) or (n_samples, n_responses)
        The training response data.

    fit_intercept: bool
        Whether or not to fit intercept, which is not penalized.

    sample_weight: None or array-like,  shape (n_samples,)
        Individual weights for each sample.

    lasso_pen_val: None, float
        The Lasso penalty value, if desired.

    ridge_pen_val: None, float
        The Lasso penalty value, if desired.

    class_weight: None, balanced
        The class weights
    solver:
        The solver for ya_glm

    ya_kws: dict
        Additional keyword arguments for the ya_glm models.

    sk_kws: dict
        Additional keyword arguments for the sklearn models.

    Output
    ------
    est, sk

    est:
        The fit ya_glm estimator

    sk:
        The fit sklearn estimator
    """
    loss = LogReg(class_weight=class_weight)

    n_samples = X.shape[0]

    if lasso_pen_val is None and ridge_pen_val is None:
        ###############
        # Unpenalized #
        ###############
        est = Vanilla(loss=loss,
                      fit_intercept=fit_intercept,
                      solver=solver,
                      **ya_kws)

        sk = LogisticRegression(fit_intercept=fit_intercept,
                                penalty='none',
                                class_weight=class_weight,
                                **sk_kws)

    elif lasso_pen_val is not None and ridge_pen_val is not None:
        ###############
        # ElasticeNet #
        ###############
        pen_val = lasso_pen_val + ridge_pen_val
        l1_ratio = lasso_pen_val / pen_val

        est = ENet(loss=loss,
                   pen_val=pen_val,
                   l1_ratio=l1_ratio,
                   fit_intercept=fit_intercept,
                   solver=solver,
                   **ya_kws)

        sk = LogisticRegression(fit_intercept=fit_intercept,
                                penalty='elasticnet',
                                class_weight=class_weight,
                                C=1/(n_samples * pen_val),
                                l1_ratio=l1_ratio,
                                solver='saga',
                                **sk_kws)

    elif lasso_pen_val is not None:
        #########
        # Lasso #
        #########

        est = Lasso(loss=loss,
                    pen_val=lasso_pen_val,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    **ya_kws)

        sk = LogisticRegression(fit_intercept=fit_intercept,
                                penalty='l1',
                                class_weight=class_weight,
                                C=1/(n_samples * lasso_pen_val),
                                solver='saga',
                                **sk_kws)

    else:
        #########
        # Ridge #
        #########
        est = Ridge(pen_val=ridge_pen_val,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    **ya_kws)

        sk = LogisticRegression(fit_intercept=fit_intercept,
                                penalty='l2',
                                class_weight=class_weight,
                                C=1/(n_samples * ridge_pen_val),
                                **sk_kws)

    est.fit(X, y)
    sk.fit(X, y)

    return est, sk
