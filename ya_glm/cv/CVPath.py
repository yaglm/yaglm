from functools import partial
import numpy as np
from joblib import Parallel, delayed
from itertools import product
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone


class CVPathMixin:
    """
    solve_glm_path
    """

    def _fit_and_score_path_getter(self, estimator):

        # get the correct fit and score method
        est = clone(estimator)
        def est_from_fit(fit_out, pre_pro_out):
            est._set_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)
            return est

        if hasattr(est, 'preprocess'):
            preprocess = est.preprocess
        else:
            preprocess = None

        fit_and_score_path = partial(score_from_fit_path,
                                     solve_path=self.solve_glm_path,
                                     est_from_fit=est_from_fit,
                                     scorer=self.cv_scorer,
                                     preprocess=preprocess)

        return fit_and_score_path

    def _run_cv(self, estimator, X, y=None, cv=None):

        # setup CV
        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        # setup path fitting function
        fit_and_score_path = self._fit_and_score_path_getter(estimator)

        cv_results, _ = \
            run_cv_path(X=X, y=y,
                        fold_iter=cv.split(X, y),
                        fit_and_score_path=fit_and_score_path,
                        kws=self._get_solve_path_kws(),
                        include_spilt_vals=True,  # TODO: maybe give option for this?
                        add_params=False,
                        n_jobs=self.cv_n_jobs,
                        verbose=self.cv_verbose,
                        pre_dispatch=self.cv_pre_dispatch)

        # add parameter sequence to CV results
        # this allows us to pass fit_path a one parameter sequence
        # while cv_results_ uses different names
        param_seq = self.get_tuning_sequence()
        cv_results = add_params_to_cv_results(param_seq=param_seq,
                                              cv_results=cv_results)

        return cv_results

    def _get_solve_path_kws(self):
        """
        The solve path function will be call as

        solve_path(X=X, y=y, **kws)

        Output
        ------
        kws: dict
            All keyword arguments for computing the path.
        """
        raise NotImplementedError


def run_cv_path(X, y, fold_iter, fit_and_score_path, kws={},
                include_spilt_vals=True, add_params=True,
                n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    """
    Runs cross-validation where we parallelize over folds and have a funtion that fits (and scores) the entire path.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The X data.

    y: array-like, shape (n_samples, )
        The response data.

    fold_iter: iterator
        Iterates over the folds. Each element should be (train, test)

    fit_and_score_path: callable(X, y, train, test, **kws) -> path_results
        Function that fits and scores the whole path and returns the results.

        path_results: dict
            path_results['train']: dict
                The training results.
            path_results['test']: dict
                The test results.
            path_results['param_seq']: list of dicts
                The path parameter values.

    kws: dict
        Key word arguments to fit_and_score_path.

    n_jobs: None, int
        Number of jobs for parallelizing over the folds.

    include_spilt_vals: bool
        Whether or not to include the split values in the cv_results

    Output
    ------
    cv_results: dict
        The cross validation results formatted to look like sklearn.model_selection.GridSerachCV.cv_results_
    """

    parallel = Parallel(n_jobs=n_jobs,
                        pre_dispatch=pre_dispatch,
                        verbose=verbose)
    fold_path_results = \
        parallel(delayed(fit_and_score_path)
                 (X=X, y=y, train=train, test=test, kws=kws)
                 for (train, test) in fold_iter)

    # TODO: maybe remove std
    cv_results, param_seq = \
        format_cv_results_from_fold_path_results(fold_path_results,
                                                 include_spilt_vals=include_spilt_vals)

    if add_params:
        cv_results = add_params_to_cv_results(param_seq=param_seq,
                                              cv_results=cv_results)
    return cv_results, param_seq


def score_from_fit_path(X, y, train, test,
                        solve_path, est_from_fit, scorer=None,
                        kws={},
                        preprocess=None):
    """

    Parameters
    -----------
    X: array-like, shape (n_samples, n_features)
        The X data.

    y: array-like, shape (n_samples, )
        The response data.

    train, test: array-like
        The train and test subsets.

    fit_path: callable(X, y, **kws)
        Function that fits the path for a given dataset.
        This should return an iterator of tuples (fit_out, param).
        Here fit_out is the output of a single fit and is used by score_from_fit
        to compute a score for each entry in the path.
        param is the parameter setting for this path value.

        kws is an (optional) set. of key word arguments to fit_path

    est_from_fit: callable(fit_out, pre_pro_out) -> estimator:
        Returnns an estimator from the output of a single fit.

    scorer: None, callable(est, X, y) -> dict or float
        Scores a single fit estimator. Shoule return either a float (which will be renamed 'score') or a dict. If None, will use est.score(X, y)

    kws: dict
        Key word arguments for fit_path

    preprocess: None, callable(X, y, copy) -> X, y, pre_pro_data
        An (optional) function that pre-processes the training data before calling the fit_path function. It should copy the data -- not modify it in place!


    Output
    ------
    path_results: dict
        path_results['train']: dict
            The training results.
        path_results['test']: dict
            The test results.
        path_results['param_seq']: list of dicts
            The path parameter values.
    """
    # TODO: add option for things that belong to fit and not train or test

    # TODO: perhaps allow scorer to be a string as in the
    #  scoring argument to GridSerachCV e.g. see
    # https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/model_selection/_search.py#L752

    # split data
    X_train = X[train, :]
    X_test = X[test, :]

    if y is not None:
        y_train = y[train]
        y_test = y[test]
    else:
        y_train = None
        y_test = None

    # possibly apply processing
    if preprocess is not None:
        # TODO: might want to handle case when y is None differently
        X_train_pro, y_train_pro, pre_pro_out = \
             preprocess(X=X_train, y=y_train, copy=True)
    else:
        X_train_pro = X_train
        y_train_pro = y_train
        pre_pro_out = {}

    # TODO: maybe add score time
    path_results = {'train': [],
                    'test': [],
                    'param_seq': [],
                    'fit_runtime': []
                    # 'score_time': []
                    }

    # Solve the path!!
    solution_path = solve_path(X=X_train_pro, y=y_train_pro, **kws)

    for fit_out, param in solution_path:

        est = est_from_fit(fit_out=fit_out, pre_pro_out=pre_pro_out)

        if scorer is None:
            # score with estimator's defualt
            tst = est.score(X=X_test, y=y_test)
            tr = est.score(X=X_train, y=y_train)

        else:
            # custom scorer
            tst = scorer(est, X=X_test, y=y_test)
            tr = scorer(est, X=X_train, y=y_train)

        path_results['test'].append(tst)
        path_results['train'].append(tr)

        path_results['param_seq'].append(param)

        if 'opt_data' in fit_out:
            if 'runtime' in fit_out['opt_data']:
                t = fit_out['opt_data']['runtime']
                path_results['fit_runtime'].append({'time': t})

    # make sure these are formatted as list of dicts
    if not isinstance(path_results['test'][0], dict):
        path_results['test'] = [{'score': val}
                                for val in path_results['test']]

        path_results['train'] = [{'score': val}
                                 for val in path_results['train']]

    if len(path_results['fit_runtime']) == 0:
        del path_results['fit_runtime']

    return path_results


def format_cv_results_from_fold_path_results(fold_path_results,
                                             include_spilt_vals=True):
    """

    Parameters
    -----------
    fold_path_results: list of dicts
        The path results for each fold. Folds are outer list.
        Each entry should be a dict, fold_res, with keys

        fold_res['test']: list of dict
            Contains the test metrics for each parameter setting

        fold_res['train']: list of dict
            Contains the train metrics for each parameter setting.
            This is optiona.

        fold_res['param_seq']: list of dicts
            The path parameter sequence. Each entry is a dict specifying the parameters for each path value.

    include_spilt_vals: bool
        Whether or not to retain the values for each split.


    Output
    ------
    cv_results: dict
        The cross validation results formatted to look like sklearn.model_selection.GridSerachCV.cv_results_

    """

    cv_results = format_cv_fold_scores([fold_res['test']
                                        for fold_res in fold_path_results],
                                       include_spilt_vals=include_spilt_vals,
                                       kind='test')
    if 'train' in fold_path_results[0]:
        tr = format_cv_fold_scores([fold_res['train']
                                    for fold_res in fold_path_results],
                                   include_spilt_vals=include_spilt_vals,
                                   kind='train')
        cv_results.update(tr)

    if 'fit_runtime' in fold_path_results[0]:
        ft = format_cv_fold_scores([fold_res['fit_runtime']
                                   for fold_res in fold_path_results],
                                   include_spilt_vals=include_spilt_vals,
                                   kind='fit')
        cv_results.update(ft)

    param_seq = fold_path_results[0]['param_seq']

    # if add_params:
    #     cv_results = add_params_to_cv_results(param_seq=param_seq,
    #                                           cv_results=cv_results)

    return cv_results, param_seq


def add_params_to_cv_results(param_seq, cv_results={}):
    """
    Adds a parameter sequence to the cv_results

    Parameters
    ----------

    param_seq: list of dicts

    cv_results: dict

    Output
    ------
    cv_results: dict

    """
    # format the parameter sequence
    param_names = list(param_seq[0].keys())
    param_dol = {name: [] for name in param_names}  # dict of lists
    for setting in param_seq:
        for param_name, value in setting.items():
            param_dol[param_name].append(value)

    # add parameter colums to cv_results
    for param_name in param_names:
        cv_results['param_{}'.format(param_name)] = \
            np.array(param_dol[param_name])

    cv_results['params'] = param_seq
    return cv_results


def format_cv_fold_scores(fold_scores, kind, include_spilt_vals=True):
    """
    Formats a list of scores

    Parameters
    ----------
    scores: list of list of dicts
        The score for each fold (outer list) for each parameter setting (inner list).

    kind: str
        Which kind of score e.g. either 'test' or 'train'

    Output
    ------
    out: dict
        out['mean_KIND_SCORENAME']: list of floats
            The mean score for this metric over the folds.
        out['std_KIND_SCORENAME']: list of floats
            The standard deviation of this metric over the folds.
        out['se_KIND_SCORENAME']: list of floats
            The standard error of this metric over the folds.

    """
    n_folds = len(fold_scores)
    n_path_vals = len(fold_scores[0])
    score_names = list(fold_scores[0][0].keys())

    out = {}
    if include_spilt_vals:
        for fold_idx, name in product(range(n_folds), score_names):
            out['split{}_{}_{}'.format(fold_idx, kind, name)] = []

    # extract fold-path data
    for name in score_names:

        out['mean_{}_{}'.format(kind, name)] = []
        out['std_{}_{}'.format(kind, name)] = []
        out['se_{}_{}'.format(kind, name)] = []

        for path_idx in range(n_path_vals):

            # values of this score for each fold
            values = [fold_scores[fold_idx][path_idx][name]
                      for fold_idx in range(n_folds)]

            _n_folds = sum(~np.isnan(values))  # exclude folds with missing values
            avg = np.nanmean(values)
            std = np.nanstd(values)
            se = std / np.sqrt(_n_folds)

            out['mean_{}_{}'.format(kind, name)].append(avg)
            out['std_{}_{}'.format(kind, name)].append(std)
            out['se_{}_{}'.format(kind, name)].append(se)

            if include_spilt_vals:
                for fold_idx in range(n_folds):
                    out['split{}_{}_{}'.format(fold_idx, kind, name)].\
                        append(values[fold_idx])

    # format into numpy
    for name in score_names:

        k = 'mean_{}_{}'.format(kind, name)
        out[k] = np.array(out[k])

        k = 'std_{}_{}'.format(kind, name)
        out[k] = np.array(out[k])

        k = 'se_{}_{}'.format(kind, name)
        out[k] = np.array(out[k])

        if include_spilt_vals:
            for fold_idx in range(n_folds):
                k = 'split{}_{}_{}'.format(fold_idx, kind, name)
                out[k] = np.array(out[k])

    return out
