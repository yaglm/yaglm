import numpy as np
import matplotlib.pyplot as plt


def plot_residuals(history, kind='primal', log=True,
                   marker='.', **kws):
    """
    Plots the primal or dual ADMM residuals.
    
    Parameters
    ----------
    history: dict
        History output by admm's opt_info. Needs tracking_level >= 2.
        
    kind: str
        Must be one of ['primal', 'dual']
        
    log: bool
        Whether or not to log the residuals.

    **kws:
        Keyword arguments to plt.plot()
    """
    
    resids = history['{}_resid'.format(kind)]
    tols = history['{}_tol'.format(kind)]
    
    if log:
        resids = np.log10(resids)
        tols = np.log10(tols)

    plt.plot(resids, marker=marker,
             color='black', label='residual',
             **kws)
    plt.plot(tols, marker=marker,
             color='grey', alpha=0.2,
             label='tolerance',
             **kws)
    plt.xlabel("Iteration")
    
    ylab = '{}'.format(kind)
    if log:
        ylab = 'log10({})'.format(ylab)
    plt.ylabel(ylab)
