import pandas as pd
import numpy as np
from copy import deepcopy


class CVSlectMixin:
    """
    cv_select_metric:

    cv_scorer:
    """
    def _get_cv_select_metric(self):

        if self.cv_select_metric is not None:
            # return cv_select_metric if provided by the user
            return self.cv_select_metric

        elif self.cv_scorer is not None:
            # if a cv_scorer is provided, try to infer the defualt score
            if hasattr(self.cv_scorer, 'default') \
                    and self.cv_scorer.default is not None:
                return self.cv_scorer.default

        # otherwise fall back on score
        return 'score'

    def _select_tune_param(self, cv_results):
        """

        Parameters
        ----------
        cv_results: dict
            Output of cross-validation results.

        Output
        ------
        best_tune_idx, best_tune_params

        best_tune_idx: int
            Index of the selected tuning parameter.

        best_tune_params: dict
            The selected tuning parameter values.
        """
        select_score = self._get_cv_select_metric()

        prefer_larger = True  # pick larger tuning parameter values among all possible candidates (mainly for 1se)

        return select_best_cv_tune_param(cv_results,
                                         rule=self.cv_select_rule,
                                         prefer_larger_param=prefer_larger,
                                         score_name=select_score)


def select_best_cv_tune_param(cv_results, rule='best',
                              prefer_larger_param=True,
                              score_name='score'):
    """
    Select the best tuning parameter index from cross-validation scores. Implmenets two rules: pick the estimator with the best fold-mean score or pick an estimator whose fold-mean score is within one standard deviation of the best score.

    Parameters
    ----------
    cv_results: dict GridSearchCV().cv_results_
        The cross-validation results.

    rule: str
        Must be one of ['score', '1se']
        'score': pick the tuning parameter with the best score.
        '1se': pick a tuning parameter within one standard error of the the best score.

    prefer_larger: bool
        Prefer larger values of the tuning parameter.

    n_folds: None, int
        Number of CV folds; needed to compute standard error.
    Output
    ------
    selected_idx: int
        Index of the selected tuning parameter.

    params_selected: dict
        The selected tuning parameters.
    """
    assert rule in ['best', '1se']

    # format the data we will need into a pd.DataFrame
    cols_we_need = ['mean_test_' + score_name, 'params']
    if rule == '1se':
        cv_results = add_se(cv_results, copy=True)
        cols_we_need.append('se_test_' + score_name)
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
            raise ValueError("1se rule not currently implemneted for multiple tuning parameters")

    # setting with the best score
    best_score = df['mean_test_' + score_name].max()

    # Candidate tuning values
    if rule == 'best':
        candidates = df.\
            query("mean_test_{} == @best_score".format(score_name))

    elif rule == '1se':

        score_se = df.\
            query("mean_test_{} == @best_score".
                  format(score_name))['se_test_{}'.format(score_name)].\
            max()

        score_lbd = best_score - score_se
        candidates = df.query("mean_test_{} >= @score_lbd".format(score_name))

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
    # get parameter dict
    # params_selected = {}
    # for k, values in cv_results.items():
    #     if k[0:6] == 'param_':
    #         param_name = k[6:]
    #         params_selected[param_name] = values[tune_idx_selected]

    return tune_idx_selected, params_selected


# def get_cv_test_results_df(cv_results, score_name='score'):
#     """
#     Returns a data frame with the cross validation results.

#     Parameters
#     ----------
#     cv_results: sklearn.model_selection.GridSearchCV().cv_results_

#     Output
#     ------
#     df: pd.DataFrame
#         The cross-validation results.

#     """
#     df = {}
#     df['params'] = cv_results['params']

#     df['mean_test_' + score_name] = cv_results['mean_test_' + score_name]
#     df['std_test_' + score_name] = cv_results['std_test_' + score_name]

#     if 'se_test_' + score_name not in cv_results.keys():
#         n_folds = _infer_n_folds(cv_results)
#         df['se_test_' + score_name] = \
#             df['std_test_' + score_name] / np.sqrt(n_folds)

#     # if there is only one tuning parameter lets put it in a column
#     param_names = list(cv_results['params'][0].keys())
#     if len(param_names) == 1:
#         param_name = param_names[0]
#         n_param_values = len(cv_results['params'])
#         df['param_values'] = [cv_results['params'][i][param_name]
#                               for i in range(n_param_values)]

#     else:
#         df['se_test_' + score_name] = cv_results['se_test_' + score_name]

#     df = pd.DataFrame(df)
#     df.index.name = 'tune_idx'

#     # tuning paramters should be unique
#     assert len(np.unique(df['param_values'])) == df.shape[0]

#     return df


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


def add_se(cv_results, copy=False):
    """
    Adds standard errors to cv results.

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
