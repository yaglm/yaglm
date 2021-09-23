import pandas as pd
from copy import deepcopy
import numpy as np


def select_tune_param(tune_results, kind, metric='score'):
    """
    Selects the best tuning parameter index.

    Parameters
    ----------
    tune_results: dict

    kind: str
        Which metric kind to use; must be one of ['train', 'test'].

    metric: str
        Which metric to use.

    Output
    ------
    idx_best: int
        The selected index.

    best_params: dict
        The selected parameters.
    """
    assert kind in ['train', 'test']
    select_key = '{}_{}'.format(kind, metric)

    # pick the best error
    values = tune_results[select_key]
    idx_best = np.argmax(values)

    best_params = tune_results['params'][idx_best]

    return idx_best, best_params


def cv_select_tune_param(cv_results, metric='score',
                         rule='best', prefer_larger_param=True):
    """
    Select the best tuning parameter index from cross-validation scores. Implmenets two rules: pick the estimator with the best fold-mean score or pick an estimator whose fold-mean score is within one standard deviation of the best score.

    Parameters
    ----------
    cv_results: dict
        The cross-validation results.

    metric: str
        Name of the test metric to use.

    rule: str
        Must be one of ['score', '1se']
        'score': pick the tuning parameter with the best score.
        '1se': pick a tuning parameter within one standard error of the the best score.

    prefer_larger_param: bool
        Prefer larger values of the tuning parameter.

    Output
    ------
    selected_idx: int
        Index of the selected tuning parameter.

    params_selected: dict
        The selected tuning parameters.
    """

    # make sure we have the requested metric
    if metric is None:
        metric = 'score'
    test_key = 'mean_test_' + metric
    if test_key not in cv_results:
        raise ValueError("{} was not found in cv_results".format(test_key))

    # check rule
    if rule not in ['best', '1se']:
        raise ValueError("rule must be one of ['best', '1se'], not {}".
                         format(rule))

    # format the data we will need into a pd.DataFrame
    cols_we_need = [test_key, 'params']
    if rule == '1se':
        se_key = 'se_test_' + metric
        cols_we_need.append(se_key)

        # if the standard errors are not in cv_results (e.g. as in sklearn)
        # then manually add them
        if se_key not in cv_results:
            cv_results = _add_se(cv_results)
    df = pd.DataFrame({c: cv_results[c] for c in cols_we_need})

    # if there is only one tuning parameter then lets pull it out
    # so we can use it for the prefer_larger
    param_names = list(cv_results['params'][0].keys())
    if len(param_names) == 1:
        single_param = True
        param_name = param_names[0]
        n_param_values = len(cv_results['params'])
        df['param_values'] = [cv_results['params'][i][param_name]
                              for i in range(n_param_values)]

    else:
        single_param = False
        if rule == '1se':
            raise NotImplementedError("1se rule not currently implemneted"
                                      "for multiple tuning parameters")

    # setting with the best score
    best_score = df['mean_test_' + metric].max()

    # Candidate tuning values
    if rule == 'best':
        candidates = df.\
            query("mean_test_{} == @best_score".format(metric))

    elif rule == '1se':

        score_se = df.\
            query("mean_test_{} == @best_score".
                  format(metric))['se_test_{}'.format(metric)].\
            max()

        score_lbd = best_score - score_se
        candidates = df.query("mean_test_{} >= @score_lbd".format(metric))

    # pick the candidate with the smallest or largest tuning value
    if single_param:
        if prefer_larger_param:
            tune_idx_selected = candidates['param_values'].idxmax()
        else:
            tune_idx_selected = candidates['param_values'].idxmin()

    else:
        # if there are multiple tuning parameters then we just
        # arbitrarily pick one candidate
        tune_idx_selected = candidates.index.values[0]

    params_selected = cv_results['params'][tune_idx_selected]

    return tune_idx_selected, params_selected


def _add_se(cv_results, copy=False):
    """
    Adds standard errors to cv results returned by sklearn that only have standard deviations.

    Parameters
    ----------
    cv_results: dict
        The cv_results

    copy: bool
        Whether or not to copy or modify in place.

    Output
    ------
    cv_results: dict
        The cv_results with standard errors added.
    """

    n_folds = None

    if copy:
        cv_results = deepcopy(cv_results)

    scores_with_std = [k for k in cv_results.keys() if k[0:4] == 'std_']
    for k in scores_with_std:
        s = k.split('_')
        s[0] = 'se'
        se_name = '_'.join(s)

        # add standard errors if they are not already present
        if se_name not in cv_results:

            if n_folds is None:
                # only compute this if we have to
                n_folds = _infer_n_folds(cv_results)

            cv_results[se_name] = np.array(cv_results[k]) / np.sqrt(n_folds)

    return cv_results


def _infer_n_folds(cv_results):
    """
    Gets the number of CV folds from a cv_results object

    Parameters
    ----------
    cv_results: dict
        The cv results.

    Output
    ------
    cv: int
        The number of cv folds
    """
    split_keys = [k for k in cv_results.keys()
                  if k.split('_')[0][0:5] == 'split']
    split_idxs = [int(k.split('_')[0].split('split')[1]) for k in split_keys]
    return max(split_idxs) + 1
