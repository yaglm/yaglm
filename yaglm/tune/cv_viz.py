import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from yaglm.cv.cv_select import add_se


def plot_cv_path(cv_results, metric, param=None, show_se=True,
                 show_train=True, selected_idx=None,
                 kind=None, log_param=False, negate=False):
    """
    Plots the cross-validation parameter path.
    """
    if kind is not None:
        show_train = False
    else:
        kind = 'test'

    if show_se:
        cv_results = add_se(cv_results, copy=True)

    # get path parameter values
    all_params = get_params(cv_results)
    if param is None:
        assert len(all_params) == 1  # try to infer parameter name
        param = all_params[0]
    assert param in all_params

    # if we dont have training results, definitely dont show them
    if 'mean_train_' + metric not in cv_results:
        show_train = False

    # get parameter values
    param_values = np.array(cv_results['param_' + param]).astype(float)
    if log_param:
        param_values = np.log10(param_values)

    # get test values
    mean_test = cv_results['mean_{}_{}'.format(kind, metric)]
    if show_se:
        # se_test = get_se(cv_results, metric, kind=kind)
        se_test = cv_results['se_{}_{}'.format(kind, metric)]

    # maybe get train values
    if show_train:
        mean_train = cv_results['mean_train_' + metric]
        if show_se:
            # se_train = get_se(cv_results, metric, kind='train')
            se_train = cv_results['se_{}_{}'.format('train', metric)]

    # maybe negate the scores
    if negate:
        mean_test *= - 1
        if show_train:
            mean_train *= - 1

    # plot!
    plt.plot(param_values, mean_test,
             marker='.', color='red', label='test')

    if show_se:
        plt.fill_between(param_values,
                         mean_test + se_test,
                         mean_test - se_test,
                         color='red', alpha=0.2)

    if show_train:

        plt.plot(param_values, mean_train,
                 marker='.', color='blue', ls='--', label='train')

        if show_se:
            plt.fill_between(param_values,
                             mean_train + se_train,
                             mean_train - se_train,
                             color='blue', alpha=0.2)

    if selected_idx:
        sel_value = param_values[selected_idx]
        if log_param:
            label = 'log10(selected) = {:1.5f}'.format(sel_value)
        else:
            label = 'selected = {:1.5f}'.format(sel_value)

        plt.axvline(sel_value, color='black',
                    label=label)

    if log_param:
        plt.xlabel('log10({})'.format(param))
    else:
        plt.xlabel(param)
    plt.ylabel(metric)

    if show_train:
        plt.legend()


def plot_cv_two_params(cv_results, param, group_param,
                       metric='score', selected_idx=None,
                       kind='test', show_se=True,
                       log_param=False, label_group_param=True,
                       group_cpal='rocket', negate=False):
    """
    Plots the cross-validation parameter path for one parameter while grouping by another parameter.
    """

    # format data
    if show_se:
        cv_results = add_se(cv_results, copy=True)

    df = pd.DataFrame(cv_results)
    group_values = np.unique(cv_results['param_' + group_param])
    group_values = np.sort(group_values)

    # set group colors
    group_colors = sns.color_palette(group_cpal, n_colors=len(group_values))

    for idx, val in enumerate(group_values):

        # pull out cv path
        this_val = df.query("{} == @val".format('param_' + group_param))
        param_vals = this_val['param_' + param]
        mean_vals = this_val['mean_{}_{}'.format(kind, metric)]

        if negate:
            mean_vals *= -1

        # maybe log
        if log_param:
            param_vals = np.log10(param_vals)

        # plot cv mean
        if label_group_param:
            label = '{} = {:1.5f}'.format(group_param, val)
        else:
            label = None

        plt.plot(param_vals, mean_vals, marker='.',
                 color=group_colors[idx],
                 label=label)

        if show_se:
            se_vals = this_val['se_{}_{}'.format(kind, metric)]

            plt.fill_between(param_vals,
                             mean_vals - se_vals,
                             mean_vals + se_vals,
                             color=group_colors[idx],
                             alpha=0.1)

    if log_param:
        plt.xlabel('log10({})'.format(param))
    else:
        plt.xlabel(param)

    plt.xlabel("{} {}".format(kind, metric))

    # posssibly label selected index
    if selected_idx:
        sel_group = df.iloc[selected_idx]['param_' + group_param]
        sel_param = df.iloc[selected_idx]['param_' + param]

        if log_param:
            sel_param = np.log10(sel_param)
            _param_name = 'log10({})'.format('param_' + param)

        else:
            _param_name = 'param_' + param

        label = '{} = {:1.5f}\n{} = {:1.5f}'.\
                format('param_' + group_param, sel_group,
                       _param_name, sel_param)

        plt.axvline(sel_param, color='black',
                    label=label)

        plt.legend()

# def get_se(cv_results, metric, kind='test'):
#     # assert kind in ['train', 'test']

#     se_key = 'se_{}_{}'.format(kind, metric)
#     if se_key in cv_results:
#         return cv_results[se_key]

#     else:
#         n_folds = _infer_n_folds(cv_results)
#         se = cv_results['std_{}_{}'.format(kind, metric)] / np.sqrt(n_folds)
#         return se


def get_params(cv_results):
    params = [k for k in cv_results.keys() if k[0:6] == 'param_']
    params = [p[6:] for p in params]
    return params
