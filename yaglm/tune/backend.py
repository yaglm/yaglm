import numpy as np
from copy import deepcopy
from itertools import product
from sklearn.metrics import get_scorer
from sklearn.utils.fixes import _joblib_parallel_args
from joblib import Parallel, delayed


def run_fit_and_score_jobs(job_configs,
                           store_ests=False, scorer=None, fit_evals=None,
                           n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    """
    Runs fit and score for a sequence of jobs.

    Parameters
    ----------
    job_configs: iterable yielding dicts
        The config for each job. Each element is a dict of keyword args to fit_and_score_from_pro excluding ['store_ests', 'scorer', 'fit_evals'].

    store_ests: bool
        Whether or not to store the fitted estimators. See fit_and_score_from_pro.

    scorer:
        See fit_and_score_from_pro.

    fit_evals:
        See fit_and_score_from_pro.

    n_jobs: None, int
        Number of jobs to run in parallel with joblib.Parallel.

    verbose: int
        Amount of printout for joblib.Parallel..

    cv_pre_dispatch: int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel execution.

    Output
    ------
    For cross-validation: cv_results

    cv_results: dict of dict of lists
        The cross-validation summary results like sklearn.model_selection.GridSearchCV().cv_results_

    For other tuning methods: results, estimators

    results: dict of dict lists
        The first level specifies the kind e.g.['fit', 'train', 'test'] and the second level is the measure e.g. 'score', 'auc', ... The second level is sorted by the tune parameter indices.

    estimators: list of Estimators
        The fitted estimators sorted by the tune parameter indices.
    """

    ######################
    # setup and run jobs #
    ######################

    par = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch,
                   **_joblib_parallel_args(prefer="threads"))

    jobs = (delayed(fit_and_score)(store_ests=store_ests,
                                   scorer=scorer,
                                   fit_evals=fit_evals,
                                   **kws) for kws in job_configs)

    output = par(jobs)

    # flatten output so it is a list of dicts
    # https://stackabuse.com/python-how-to-flatten-list-of-lists/
    output = [item for sublist in output for item in sublist]

    # format results
    if 'fold_idx' in output[0]:
        # cross-validation results
        return get_cv_results(output=output,
                              include_spilt_vals=True,
                              include_std=False,  # differs from sklearn
                              include_se=True,
                              include_params=True)
    else:
        # otherwise
        results, estimators = get_tune_output_dol(output, include_params=True)
        results = format_tune_scores(results)

        if store_ests:
            return results, estimators
        else:
            return results, None


def get_cross_validation_jobs(raw_data, est, solver, tune_iter, fold_iter,
                              path_algo=True, solver_init={}):
    """
    Iterates over all jobs for cross-validation with a double loop. The outer loop splits and processes each fold; the inner loop is over the parameter settings.

    Parameters
    ----------
    raw_data: dict
        A dict containing the raw data with keys ['X', 'y', 'sample_weights'].

    est: Glm
        The base estimator. This is used for preprocessing the CV training data and for computing evaluation metrics.

    solver: GlmSolver
        The solver to use.

    tune_iter:
        An object that iterates over all the tuning parameter settings e.g. PenaltyPerLossFlavorTuner().

    fold_iter: iterable
        Iterates over the cv folds e.g. the output of cv.split(X=X, y=y).

    path_algo: bool
        Whether or not to use the solver's path algorithm if it has one available.

    solver_init: dict
        Initialization for the solver.

    Yields
    ------
    job_configs: dict
        Keyword arguments that will be passed to fit_and_score() through run_fit_and_score_jobs(). If a path algorithm is used, each dict represents one fold/path parameter setting. If a path algorithm is not used this represents one fold and one parameter setting.
    """

    # use a path algo if the solver has one available
    path_algo = path_algo and solver.has_path_algo

    # outer loop over folds, inner loop over parameter settings
    for fold_idx, (train, test) in enumerate(fold_iter):

        # split/process train data
        solver_data, eval_data = \
            split_and_process(**raw_data,  # X, y, sample_weight
                              est=est, train=train, test=test)

        # setup tuning prameter iterator
        if path_algo:
            config_iter = tune_iter.iter_configs_with_pen_path(with_params=True)
        else:
            config_iter = tune_iter.iter_configs(with_params=True)

        for tune_idx_outer, tune_configs in enumerate(config_iter):

            yield {'solver_data': solver_data,
                   'solver': solver,
                   'tune_configs': tune_configs,
                   'solver_init': solver_init,

                   'path_algo': path_algo,
                   'tune_idx_outer': tune_idx_outer,
                   'fold_idx': fold_idx,  # track which CV fold this is

                   **eval_data
                   }


def get_validation_jobs(raw_data, est, solver, tune_iter,
                        train, test,
                        path_algo=True,
                        solver_init={}):
    """
    Iterates over all jobs for tuning with a validation set.

    Parameters
    ----------
    raw_data: dict
        A dict containing the raw data with keys ['X', 'y', 'sample_weights'].

    est: Glm
        The base estimator. This is used for preprocessing the CV training data and for computing evaluation metrics.

    solver: GlmSolver
        The solver to use.

    tune_iter:
        An object that iterates over all the tuning parameter settings e.g. PenaltyPerLossFlavorTuner().

    train, test: array-like of ints
        The train/test indices.

    path_algo: bool
        Whether or not to use the solver's path algorithm if it has one available.

    solver_init: dict
        Initialization for the solver.

    Yields
    ------
    job_configs: dict
        Keyword arguments that will be passed to fit_and_score() through run_fit_and_score_jobs(). If a path algorithm is used, each dict represents one path parameter setting. If a path algorithm is not used this represents one parameter setting.
    """

    # use a path algo if the solver has one available
    path_algo = path_algo and solver.has_path_algo

    # split/process train data
    solver_data, eval_data = split_and_process(**raw_data,  # X,y,sample_weight
                                               est=est,
                                               train=train, test=test)

    # setup tuning prameter iterator
    if path_algo:
        config_iter = tune_iter.iter_configs_with_pen_path(with_params=True)
    else:
        config_iter = tune_iter.iter_configs(with_params=True)

    for tune_idx_outer, tune_configs in enumerate(config_iter):

        yield {'solver_data': solver_data,
               'solver': solver,
               'tune_configs': tune_configs,
               'solver_init': solver_init,

               'path_algo': path_algo,
               'tune_idx_outer': tune_idx_outer,

               **eval_data
               }


def get_train_jobs(pro_data, raw_data, pre_pro_out,
                   est, solver, tune_iter,
                   path_algo=True,
                   solver_init={}):
    """
    Iterates over all jobs for training only tuning.

    Parameters
    ----------
    pro_data: dict
        A dict containing the processed data with keys ['X', 'y', 'sample_weights'].

    raw_data: dict
        A dict containing the raw data with keys ['X', 'y', 'sample_weights'].

    pre_pro_out: dict
        The preprocessing output of the training data.

    est: Glm
        The base estimator. This is used for preprocessing the CV training data and for computing evaluation metrics.

    solver: GlmSolver
        The solver to use.

    tune_iter:
        An object that iterates over all the tuning parameter settings e.g. PenaltyPerLossFlavorTuner().

    path_algo: bool
        Whether or not to use the solver's path algorithm if it has one available.

    solver_init: dict
        Initialization for the solver.

    Yields
    ------
    job_configs: dict
        Keyword arguments that will be passed to fit_and_score() through run_fit_and_score_jobs(). If a path algorithm is used, each dict represents one path parameter setting. If a path algorithm is not used this represents one parameter setting.
    """

    # processed data to be passed to the solver
    solver_data = {**pro_data,
                   # TODO: bit of an awkward place to put this
                   'fit_intercept': est.fit_intercept
                   }

    # setup evaluation data
    eval_data = {'pre_pro_out': pre_pro_out, 'base_estimator': est}
    for k in raw_data.keys():
        eval_data[k + '_train'] = raw_data[k]

    # use a path algo if the solver has one available
    path_algo = path_algo and solver.has_path_algo

    # setup tuning prameter iterator
    if path_algo:
        config_iter = tune_iter.iter_configs_with_pen_path(with_params=True)
    else:
        config_iter = tune_iter.iter_configs(with_params=True)

    for tune_idx_outer, tune_configs in enumerate(config_iter):

        yield {'solver_data': solver_data,
               'solver': solver,
               'tune_configs': tune_configs,
               'solver_init': solver_init,

               'path_algo': path_algo,
               'tune_idx_outer': tune_idx_outer,

               **eval_data
               }


def split_and_process(X, y, est, train=None, test=None, sample_weight=None):
    """
    Possibly splits the data into train/test sets then processes the training data.

    Parameters
    ----------
    X: array-like
        The covariate data.

    y: array-like
        The response data.

    est: Estimator
        The base estimator, used for preprocessing.

    train: None, array-like
        (Optional) Indices for training samples.

    test: None, array-like
        (Optional) Indices for test samples.

    fit_params: None, array-like
        (Optional) Sample weights.

    Output
    ------
    solver_data, eval_data

    solver_data: dict
        The processed data that will be sent to the solver. Includes keys

        X, y: array-like
            The processed training data.

        sample_weight: None, array-like
            The processed training sample weights.

    eval_data: dict
        The raw data that will be used for computing evaluation metrics. Includes keys

        X_train, y_train: array-like
            The raw training data.

        pre_pro_out: dict
            The preprocessing output for the training data.

        base_estimator: est
            The base estimator used for fitting.

        X_test, y_test: array-like
            The raw test data.

        fit_params_test:
            The test fit params.
    """

    ##########################
    # extract training data #
    #########################
    if train is None:
        X_train, y_train, sample_weight_train = X, y, sample_weight

    else:
        X_train = X[train, :]
        y_train = y[train]

        if sample_weight is not None:
            sample_weight_train = sample_weight[train]
        else:
            sample_weight_train = None

    #########################
    # process training data #
    #########################

    # TODO: need to think carefully about processing fit_params_train
    pro_data, pre_pro_out = \
        est.preprocess(X=X_train, y=y_train,
                       sample_weight=sample_weight_train,
                       copy=True)

    # processed data to be passed to the solver
    solver_data = {**pro_data,

                   # TODO: bit of an awkward place to put this
                   'fit_intercept': est.fit_intercept
                   }

    # raw data that will be used for evaulation
    eval_data = {'X_train': X_train,
                 'y_train': y_train,
                 'sample_weight_train': sample_weight_train,
                 'pre_pro_out': pre_pro_out,
                 'base_estimator': est  # TODO: clone here?
                 }
    # TODO: intercept_init, coef_init, sample_weight, penalty_data

    #####################
    # extract test data #
    #####################
    if test is not None:
        eval_data['X_test'] = X[test, :]
        eval_data['y_test'] = y[test]

        if sample_weight is not None:
            eval_data['sample_weight_test'] = sample_weight[test]
        else:
            eval_data['sample_weight_test'] = None

    return solver_data, eval_data


# TODO: add store best estimator only functionality
def fit_and_score(solver_data, solver, path_algo, solver_init,
                  tune_configs, tune_idx_outer,

                  base_estimator, pre_pro_out,
                  X_train, y_train,
                  X_test=None, y_test=None,
                  sample_weight_train=None,
                  sample_weight_test=None,

                  fold_idx=None,
                  store_ests=False,
                  scorer=None,
                  fit_evals=None):
    """
    Fits and scores an estimator for either a single parameter setting or a path of parameters.

    Parameters
    -----------
    solver_data: dict
        The input to solver.setup() excluding the config arguments e.g. has keys ['X', 'y', 'sample_weight', 'fit_intercept'].

    solver: GlmSolver
        The solver object that computes either a single solution of solution path.

    path_algo: bool
        Whether or not to use the solver's path algorithm if it has one available.

    solver_init: dict
        Initialization for the solver's algorithm see solver.solve() or solver.solve_penalty_path()

    tune_configs: tuple
        For path algorithms this is a 3 tuple (configs, single_param_settings, penalty_path).

        configs: dict
            A dict containing the loss, penalty and constraint config objects.

        single_param_settings: dict
            The parameter settings all parameter not included in the path.

        penalty_path: list of dicts
            The penalty parameter path.

        For non-path algorithms this is a 2 tuple (configs, tuned_params)

        configs: dict
            A dict containing the loss, penalty and constraint config objects for this tuning parameter setting.

        tuned_params: dict
            The values of the tuning parameters set here.

    tune_idx_outer: int
        The outer index of this tuning parameter setting. This only differs from the returned tune_idx_outer if the param settings contain path parameter settings.

    base_estimator: estimator
        The base estimator we will use to set the fit from the solution. We also use the inferencer object stored in this base_estimator if it has one e.g. for computing the degrees of freedom.

    pre_pro_out: dict
        The pre-processing output need by base_estimator._set_fit.

    X_train: array-like, shape (n_samples_train, n_features)
        The raw X training data.

    y_train: array-like, shape (n_samples_train, )
        The raw y response training data.

    X_test: array-like, shape (n_samples_test, n_features)
        The X test data.

    y_test: array-like, shape (n_samples_train, )
        The y test response data.

    sample_weight_train: None, array-like (n_samples_train, )
            (Optional) Train sample weights.

    sample_weight_test: None, array-like (n_samples_test, )
            (Optional) Test sample weights.

    fold_idx: None, int
        (Optional) Index of the cross-validation fold.

    store_ests: float
        Whether or not to store and return the fit estimators.

    scorer: None, str, callable(est, X, y) -> dict or float
        Scores a single fit estimator. Should return either a float (which will be renamed 'score') or a dict. If None, will use est.score(X, y). If is a string then will be the scorer from sklearn.metrics.get_scorer

    fit_evals: None, callable(est) -> dict or float
        (Optional) function that computes quantities from the fit estimator e.g. np.linalg.norm(est.coef_).

    Output
    ------
    results: dict
        Results.

    results['tune_idxs']: list of ints
        Indices of the parameters used here.

    results['params']: list of dicts
        The tuning parameter values.

    results['train']: list of dicts
        The training scores for each parameter setting.

    results['test']: list of dicts
        The test scores for each parameter setting.

    results['fit']: list of dicts
        The fit evaluation measures.

    results['est']: list of estimators
        (Optional) The fit estimators. Included only if store_ests=True.

    results['fold_idx']: list of estimators
        (Optional) The cross-validation fold index. Included if fold_idx is not None.
    """

    #################################
    # Solve optimization problem(s) #
    #################################
    # TODO-ADD: we can avoid some repeated computation by more intelligently
    # doing solver.setup() before calling fit_and_score().
    solver_init = {} if solver_init is None else solver_init

    if path_algo:

        # solve!
        # note solve_penalty_path may return a generator in which case
        # the solutions are actually computed below
        configs, single_param_settings, penalty_path = tune_configs

        # defensive copy since this gets used below, but can be
        # accidently modified in place by the solver
        base_configs = deepcopy(configs)

        solver.setup(**solver_data, **configs)
        solutions = solver.solve_penalty_path(penalty_path=penalty_path,
                                              **solver_init)

        # formatting
        # get uniuqe path tuning parameter settings
        tuned_params = []
        for pen_path_params in penalty_path:

            # copy the single_param_settings
            this_param_settings = {**single_param_settings}

            # add penalty path settings
            if 'penalty' in this_param_settings:
                this_param_settings['penalty'].update(pen_path_params)
            else:
                this_param_settings['penalty'] = pen_path_params

            tuned_params.append(this_param_settings)

    else:
        # solve!
        configs, tuned_params = tune_configs

        # defensive copy since this gets used below, but can be
        # accidently modified in place by the solver
        base_configs = deepcopy(configs)

        solver.setup(**solver_data, **configs)
        solutions = solver.solve(**solver_init)

        # format!
        solutions = [solutions]
        tuned_params = [tuned_params]

    # reformated tuned param from list of dict of dicts to just list of dicts
    kinds = tuned_params[0].keys()
    for i in range(len(tuned_params)):
        old_params = tuned_params[i]
        new_params = {}

        for kind in kinds:
            if kind in old_params:
                for param_name, value in old_params[kind].items():
                    new_key = kind + '__' + param_name
                    new_params[new_key] = value

        tuned_params[i] = new_params

    #######################
    # score each soluiton #
    #######################

    results = []
    # if solve_penalty_path returned a generator the solutions are actually computed here
    for tune_idx_inner, soln_out in enumerate(solutions):

        res = {'tune_idx_inner': tune_idx_inner,
               'tune_idx_outer': tune_idx_outer,
               'params': tuned_params[tune_idx_inner]}

        fit_out, _, opt_info = soln_out

        #######################
        # setup fit estimator #
        #######################

        # get configs for base estimator
        if path_algo and len(tuned_params[tune_idx_inner]) > 0:
            # set the penalty config to have this path elements's value
            penalty_params = {}
            for key, value in tuned_params[tune_idx_inner].items():

                # pull out the kind of this parameter (penalty, loss, constraint, flavot)
                # the keys are formatted as KIND__p1__p2__....

                splt = key.split('__')
                KIND = splt[0]
                name = '__'.join(splt[1:])  # name of the parameter

                # set either penalty parameters or flavor parameters
                # the flavor parameter's name should be relarive to the
                # penalty's set_prams() function.
                if KIND in ['penalty', 'flavor']:
                    penalty_params[name] = value

            base_configs['penalty'].set_params(**penalty_params)

        # set fit coef/intercept for base estimator
        base_estimator._set_fit(fit_out=fit_out,
                                pre_pro_out=pre_pro_out,
                                configs=base_configs)

        # run any statistical inference
        # TODO: be careful about passing the raw vs. processed data here
        base_estimator.\
            run_after_fit_inference(X=X_train, y=y_train,
                                    sample_weight=sample_weight_train)

        ###########################
        # score the fit estimator #
        ###########################
        
        # TODO: add sample weight and other fit params
        tst = None
        if scorer is None:  # score with estimator's defualt

            # train score
            tr = base_estimator.score(X=X_train, y=y_train,
                                      sample_weight=sample_weight_train)
            
            # (optional) test score
            if X_test is not None:
                tst = base_estimator.score(X=X_test, y=y_test,
                                           sample_weight=sample_weight_test)

        else:  # custom scorer

            # possibly get scoring from sklearn

            if type(scorer) == str:
                # default_score_name = scorer
                scorer = get_scorer(scorer)

            # train score
            tr = scorer(base_estimator, X_train, y_train,  # X=, y_true=
                        sample_weight=sample_weight_train)
            
            # test score
            if X_test is not None:
                tst = scorer(base_estimator, X_test, y_test,
                             sample_weight=sample_weight_test)

        # make sure we have dict formatting
        if not isinstance(tr, dict):
            tr = {'score': tr}
        if tst is not None and not isinstance(tst, dict):
            tst = {'score': tst}
        
        res['train'] = tr
        res['test'] = tst

        ###########################
        # fit evaluation measures #
        ###########################
        fit = {}
        if fit_evals is not None:
            fit.update(fit_evals(base_estimator))
        
        if 'runtime' in opt_info:
            # get solver runtime from fit_out if available 
            fit['runtime'] = opt_info['runtime']
            
        res['fit'] = fit

        #########################
        # maybe save estimators #
        #########################
        
        if store_ests:
            res['est'] = deepcopy(base_estimator)
        else:
            res['est'] = None

        # maybe add fold idx to results
        if fold_idx is not None:
            res['fold_idx'] = fold_idx

        results.append(res)

    return results


def format_tune_scores(results):
    """
    Creates a tune_results dict from the output of get_tune_output_dol applied to the results output by fit_and_score.

    This creates the proper score names e.g. train_score, test_score, etc. It also formats the tune parameter list.

    Parameters
    ----------
    results: dict of lists
        The dict of list version of the results from fit_and_score()

    Output
    ------
    results; dict of lists
        The formated results.

    """
    new_results = {}

    if 'params' in results:
        new_results['params'] = results.pop('params')

    kinds = ['train', 'test', 'fit']
    for kind in kinds:
        for metric, value in results[kind].items():
            new_key = '{}_{}'.format(kind, metric)
            new_results[new_key] = value

    return new_results


def get_tune_output_dol(output, include_params=True):
    """
    Takes the output of fit_and_score_jobs that is a list of dicts and converts it to a dict of lists.

    Parameters
    ----------
    output: list of dicts
        The fit and score output.

    include_params: bool
        Whether or not to include the parameter sequence in the rsults.

    Output
    ------
    results, estimators

    results: dict of dict lists
        The first level specifies the kind e.g.['fit', 'train', 'test'] and the second level is the measure e.g. 'score', 'auc', ... The second level is sorted by the tune parameter indices.

    estimators: list of Estimators
        The fitted estimators sorted by the tune parameter indices.
    """

    kinds = ['train', 'test', 'fit']

    results = {k: {} for k in kinds}  # dict of lists results
    estimators = []  # list of fitted estimators
    param_seq = []  # list of parameters

    # order of parameters
    tune_idxs_outer = []
    tune_idxs_inner = []

    for res in output:
        tune_idxs_outer.append(res['tune_idx_outer'])
        tune_idxs_inner.append(res['tune_idx_inner'])

        estimators.append(res['est'])
        param_seq.append(res['params'])

        for kind in kinds:

            # if there are no results for this kind then skip
            if res[kind] is None:
                continue

            # extract all keys for this kind
            for key, value in res[kind].items():

                # if this is the first time we've seen this key
                # add it to the results dict
                if key not in results[kind].keys():
                    results[kind][key] = []

                results[kind][key].append(value)

    # make sure everything is sorted according to parameter indices
    # sort lexographically primarily by outer idxs then by inner idxs
    sort_idxs = np.lexsort(keys=(tune_idxs_inner, tune_idxs_outer))
    estimators = [estimators[i] for i in sort_idxs]
    param_seq = [param_seq[i] for i in sort_idxs]
    for kind in kinds:
        for key in results[kind].keys():
            results[kind][key] = [results[kind][key][i] for i in sort_idxs]

    if include_params:
        results['params'] = param_seq

    return results, estimators


def get_cv_results(output, include_spilt_vals=True,
                   include_std=True,
                   include_se=True,
                   include_params=True):
    """
    Takes the output of fit_and_score_jobs and formats it into a cv_results dict that is like sklearn.model_selection.GridSearchCV().cv_results_

    Parameters
    ----------
    output: list of dicts
        The fit and score output.

    include_spilt_vals: bool
        Whether or not to include the raw values for each split.

    include_std: bool
        Whether or not to include the standard deviation for the fold results.

    include_se: bool
        Whether or not to include the standard error for the fold results.

    include_params: bool
        Whether or not to include the tune parameter settings in the cv_results dict.

    Output
    ------
    """
    # organize results by fold
    fold_results = {}
    for res in output:
        fold_idx = res.pop('fold_idx')

        if fold_idx in fold_results.keys():
            fold_results[fold_idx].append(res)

        else:
            fold_results[fold_idx] = [res]

    # process each fold's results
    fold_idxs_sorted = np.sort(list(fold_results.keys()))
    for fold_idx in fold_idxs_sorted:
        # note we drop the fitted estimators here
        fold_results[fold_idx], _ \
             = get_tune_output_dol(fold_results[fold_idx],
                                   include_params=include_params)

    # compute summary statistics (e.g. mean, std, se) for fold results
    cv_results = {}
    present_kinds = fold_results[fold_idxs_sorted[0]].keys()
    for kind in ['train', 'test', 'fit']:

        # skip if we dont have anything for this kind
        if kind not in present_kinds:
            continue

        # fold scores for this kind e.g. all train scores
        # list of dict of lists
        fold_scores = [fold_results[fold_idx][kind]
                       for fold_idx in fold_idxs_sorted]

        # dict of lists
        # each dict entry is a KIND_METRIC e.g. train_score
        # and the entries of the list are the summary for each
        # tuning parameter value e.g. mean_train_score
        summary = format_cv_fold_scores(fold_scores=fold_scores,
                                        kind=kind,
                                        include_spilt_vals=include_spilt_vals,
                                        include_std=include_std,
                                        include_se=include_se
                                        )

        cv_results.update(summary)

    # possibly add tune parameters
    if include_params:
        cv_results['params'] = fold_results[fold_idxs_sorted[0]]['params']

    return cv_results


def format_cv_fold_scores(fold_scores, kind,
                          include_spilt_vals=True,
                          include_std=True,
                          include_se=True):
    """
    Formats cross-validation results e.g computes mean, std, se of each fold.

    Parameters
    ----------
    scores: list of dict of lists
        The score for each fold (outer list) for each parameter setting (inner list).

    kind: str
        Which kind of score e.g. either 'test' or 'train'

    include_spilt_vals: bool
        Whether or not to include the raw values for each split.

    include_std: bool
        Whether or not to include the standard deviation for the fold results.

    include_se: bool
        Whether or not to include the standard error for the fold results.

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
    score_names = list(fold_scores[0].keys())
    n_tune_vals = len(fold_scores[0][score_names[0]])

    # format output
    out = {}
    if include_spilt_vals:
        for fold_idx, name in product(range(n_folds), score_names):
            out['split{}_{}_{}'.format(fold_idx, kind, name)] = []

    # extract fold-tune setting data
    for name in score_names:

        out['mean_{}_{}'.format(kind, name)] = []

        if include_std:
            out['std_{}_{}'.format(kind, name)] = []

        if include_se:
            out['se_{}_{}'.format(kind, name)] = []

        for tune_idx in range(n_tune_vals):

            # values of this score for each fold
            values = [fold_scores[fold_idx][name][tune_idx]
                      for fold_idx in range(n_folds)]

            # compute summary statistics
            _n_folds = sum(~np.isnan(values))  # exclude folds with missing values
            avg = np.nanmean(values)
            std = np.nanstd(values)
            se = std / np.sqrt(_n_folds)

            # store summary stats
            out['mean_{}_{}'.format(kind, name)].append(avg)

            if include_std:
                out['std_{}_{}'.format(kind, name)].append(std)

            if include_se:
                out['se_{}_{}'.format(kind, name)].append(se)

            # possibly include values
            if include_spilt_vals:
                for fold_idx in range(n_folds):
                    k = 'split{}_{}_{}'.format(fold_idx, kind, name)
                    out[k].append(values[fold_idx])

    # format into numpy
    for name in score_names:

        k = 'mean_{}_{}'.format(kind, name)
        out[k] = np.array(out[k])

        if include_std:
            k = 'std_{}_{}'.format(kind, name)
            out[k] = np.array(out[k])

        if include_se:
            k = 'se_{}_{}'.format(kind, name)
            out[k] = np.array(out[k])

        if include_spilt_vals:
            for fold_idx in range(n_folds):
                k = 'split{}_{}_{}'.format(fold_idx, kind, name)

                out[k] = np.array(out[k])

    return out
