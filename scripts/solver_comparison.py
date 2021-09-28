from time import time
from itertools import combinations
import pandas as pd
import os
from os.path import join
import argparse
import numpy as np
from itertools import product

from ya_glm.config.loss import LinReg, Poisson, Quantile, Huber
from ya_glm.config.penalty import Ridge, NoPenalty, GeneralizedRidge, \
    Lasso, GroupLasso, \
    FusedLasso, GeneralizedLasso,  \
    SeparableSum, OverlappingSum

from ya_glm.toy_data import sample_sparse_lin_reg, sample_sparse_poisson_reg
from ya_glm.solver.FISTA import FISTA
from ya_glm.solver.Cvxpy import Cvxpy
from ya_glm.solver.ZhuADMM import ZhuADMM

from ya_glm.trend_filtering import get_tf1

parser = argparse.\
    ArgumentParser(description="Compare solvers for various "
                               "loss + penalty combinations.")

parser.add_argument('--n_samples', default=100, type=int,
                    help='Number of samples.')


parser.add_argument('--n_features', default=10, type=int,
                    help='Number of features.')


parser.add_argument('--save_dir', default=None,
                    help='Directory where we (optionally) save the output.')

args = parser.parse_args()

# TODO: remove
args.save_dir = '/Users/iaincarmichael/Dropbox/Research/ya_glm/notebooks/temp_results'


def get_diffs(v1, v2):
    """Measures of distance between v1 and v2"""
    value = np.array(v1 - v2).reshape(-1)
    return {'L1': abs(value).sum(),
            'max': max(value),
            'mad': abs(value).mean(),  # mean absolute diff
            'rmse': np.sqrt(np.mean(value**2))  # root mean sq err
            }


#########
# setup #
#########

solvers = {'fista': FISTA(),
           'admm': ZhuADMM(),
           # 'admm_long': ZhuADMM(rtol=1e-7, atol=1e-7, max_iter=1e4),
           'cvxpy': Cvxpy()}


losses = {'lin_reg': LinReg(),
          'poisson': Poisson(),
          'quantile': Quantile(),
          'huber': Huber()
          }

n_features = args.n_features
mat = get_tf1(n_features)
weights = np.linspace(0, 1, num=n_features)
groups = [range(n_features // 2), range(n_features // 2, n_features)]

penalties = {'no_penalty': NoPenalty(),
             'lasso': Lasso(pen_val=.1),
             'weighted_lasso': Lasso(pen_val=.1, weights=weights),
             'ridge': Ridge(),
             'gen_ridge': GeneralizedRidge(mat=mat),
             'group_lasso': GroupLasso(pen_val=.1, groups=groups),
             # 'exclusive_group': ExclusiveGroupLasso(),
             'fused_lasso': FusedLasso(edgelist='chain'),
             'gen_lasso': GeneralizedLasso(mat=mat),
             # 'enet': ElasticNet(mix_val=0.5),

             'sep_sum': SeparableSum(groups={'first': groups[0],
                                             'second': groups[1]},
                                     first=NoPenalty(),
                                     second=Lasso(pen_val=.1)),

             'overlapping_sum': OverlappingSum(ridge=Ridge(),
                                               lasso=Lasso(pen_val=.1))

             }

samplers = {'lin_reg': sample_sparse_lin_reg,
            'poisson': sample_sparse_poisson_reg,
            'quantile': sample_sparse_lin_reg,
            'huber': sample_sparse_lin_reg}


sampler_kws = {'n_samples': args.n_samples,
               'n_features': args.n_features,
               'random_state': 0}

fit_intercept = True

#####################
# solve and compare #
#####################
runtimes = []  # track solver runtimes
diffs = []  # track solution differences
for loss_name, loss in losses.items():

    # sample data
    X, y, _ = samplers[loss_name](**sampler_kws)

    for pen_name, penalty in penalties.items():
        print('\n')

        lp_identif = {'loss': loss_name, 'penalty': pen_name}

        setup_kws = {'X': X, 'y': y,
                     'loss': loss, 'penalty': penalty,
                     'fit_intercept': fit_intercept}

        solver_solns = {}
        solver_runtimes = {}
        for solver_name, solver in solvers.items():

            # setup and run solver if it is applicable to this problem
            if solver.is_applicable(loss=loss, penalty=penalty):

                try:
                    start_time = time()
                    solver.setup(**setup_kws)
                    s = solver.solve()[0]['coef']
                    t = time() - start_time
                except:
                    print('{} failed!'.format(solver_name))
                    s, t = None, None

            else:
                s, t = None, None

            # track data
            solver_solns[solver_name] = s
            runtimes.append({**lp_identif, 'solver': solver_name,
                             'runtime': t})

            if t is None:
                print('{} {} not solvable by {}'.
                      format(loss_name, pen_name, solver_name))
            else:
                soln_l1 = abs(s).sum()
                print('{} {} {} took {:1.4f} seconds, '
                      'soln L1 = {:1.4f}'.
                      format(loss_name, pen_name, solver_name, t, soln_l1))

        # compute difference between each solvers' solution
        solver_diffs = {}
        for (k1, v1), (k2, v2) in combinations(solver_solns.items(), 2):
            if v1 is None or v2 is None:
                diff_metrics = {}
            else:
                diff_metrics = get_diffs(v1, v2)
            # store data
            diff_name = k1 + '__vs__' + k2
            diffs.append({**lp_identif, 'comparison': diff_name,
                          **diff_metrics})

###################
# format results  #
###################

# convert to pandas and sort
diffs = pd.DataFrame(diffs)
for (loss, penalty), df in diffs.groupby(['loss', 'penalty']):
    diffs.loc[df.index, :] = df.sort_values('L1').values

diffs['big'] = diffs['max'] > 0.1  # mark big differences

# convert to pandas and sort
runtimes = pd.DataFrame(runtimes)

for (loss, penalty), df in runtimes.groupby(['loss', 'penalty']):
    runtimes.loc[df.index, :] = df.sort_values('runtime').values

#################
# Print results #
#################
print('n_samples = {}, n_features = {}'.format(args.n_samples, args.n_features))
print('\n\n\n----------Runtimes----------\n\n')

# print runtimes
for (loss, penalty) in product(losses.keys(), penalties.keys()):
    df = runtimes.query("loss == @loss & penalty == @penalty")
    print(loss, penalty)
    print(df.to_string(index=False))
    print('\n')

print('\n\n\n----------Solution comparisons----------')

# print diffs
for (loss, penalty) in product(losses.keys(), penalties.keys()):
    df = diffs.query("loss == @loss & penalty == @penalty")

    print(loss, penalty)
    for _, row in df.iterrows():
        s1, s2 = row['comparison'].split('__vs__')

        to_print = '{} vs. {} L1 = {}, max = {}'.\
            format(s1, s2, row['L1'], row['max'])

        if row['big']:
            to_print = '*' + to_print
        print(to_print)

    print('\n\n')

# possibly save results
if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)
    name_stub = 'n={}_d={}'.format(args.n_samples, args.n_features)

    runtimes.to_csv(join(args.save_dir, '{}_runtimes.csv'.format(name_stub)))
    diffs.to_csv(join(args.save_dir, '{}_diffs.csv'.format(name_stub)))
