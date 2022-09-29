import matplotlib.pyplot as plt
import numpy as np


def plot_compare(data_xs, data_exp, data_exp_err, data_pred, yscale='linear', figname=None, figsize=[10., 8.], axsettings={}, **kwargs):
    """
    Plot the experimental data with error bars and compare to the theoretical prediction.

    Args:
        data_x: the x axis with the same length as the data
        data_exp: 1-D array-like with shape (N)
            This is the experimental data.
        data_exp_err: 1-D array-like with shape (N)
            This is the uncertainty of experimental data.
        data_pred: 1-D array-like with shape (N)
            This is the theoretical prediction
        yscale: string. default: 'linear'
            It determines scale of y axis
            Possible values: 'linear' and 'log'; see matplotlib documentation for more
        figname: string. default:None
            If not None, then a figure will be saved with the name being `figname`
        figsize: a tuple of floats. Default: [10., 8.]
            The size of the figure
        axsettings: dictionary. Default: empty
            Settings that can be passed to customize the setting of plot.
            Possible keys: 'xlabel', 'ylabel', 'title', 'xscale', 'yscale', etc.
        **kwargs: extra keyword arguments
            These arguments will be passed to the plot of the error bars.
                e.g. if you use plot_compare(......., capsize=2), then capsize=2 will be
                part of the kwargs. And this will be passed to matplotlib's errorbar function
                e.g. if you use plot_compare(......., capsize=2, ecolor='blue'), then both capsize=2 and ecolor='blue' will be
                part of the kwargs. And this will be passed to matplotlib's errorbar function
            Possible options:
                capsize: float, the length of error bar cap
                ecolor: color of error bar lines
                elinewidth: width of error bar lines
                color: color
                linestyle: style of the line

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)

    axsettings['yscale']=yscale
    ax.set(**axsettings) 

    #xs = np.arange(len(data_exp))
    xs = data_xs
    
    ax.errorbar(xs, data_exp, yerr = data_exp_err, label='data', **kwargs)
    ax.plot(xs, data_pred, label='prediction')
    ax.legend()    

    if figname:
        fig.savefig(figname, dpi=240, bbox_inches='tight')

    plt.show()



