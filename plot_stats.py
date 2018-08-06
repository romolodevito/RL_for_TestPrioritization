""" Plots RL stats """
from __future__ import division

import glob
import os

import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stats import plot_result_difference_bars

try:
    import cPickle as pickle
except:
    import pickle


def plot_stats(prefix, stats_file, val_file, mean_interval=1, plot_graphs=True, save_graphs=False):
    plot_stats_single_figure(prefix, stats_file, val_file, mean_interval, plot_graphs, save_graphs)


## Plots a single figure for Recall and a separate one for the NAPFD . 
## Both the difference bars and the line graphs are present in a figure.
## For the difference bar, the line y=0 is the reference with respect to the algorithm . In other words, the graph of the difference of the performance is plotted.
## Positive differences show better performance of the comparison algorithma nd negative difference show better performance of RETECS.
def plot_stats_single_figure(prefix, stats_file, val_file, mean_interval=1, plot_graphs=True, save_graphs=False):
    if not plot_graphs and not save_graphs:
        print('Set at least one of plot_graphs and save_graphs to True')
        return

    sns.set_style('whitegrid')

    ## Opening the Stats File
    stats = pickle.load(open(stats_file, 'rb'))

    ## Suplots 2 means that 2 sub-graphs will be plotted in a graph.
    ## One will be the difference graph and the other one will be the simple value graph.
    fig, ax = plt.subplots(2)
    (qax, rax) = ax

    fig,ax2 = plt.subplots(2)
    (qax2,rax2) =ax2

    failure_count = np.add(stats['detected'], stats['missed'])

    ## Range of X will be determined by the num of scenarios i.e the CI cycles
    x = range(1, int(len(stats['scenarios']) / mean_interval) + 1)
    perc_missed = [m / fc if fc > 0 else 0 for (m, fc) in zip(stats['missed'], failure_count)]
    mean_missed, missed_fit = mean_values(x, perc_missed, mean_interval)
    mean_reward, reward_fit = mean_values(x, stats['rewards'], mean_interval)

    plot_results_line_graph(stats, 'napfd', mean_interval, qax, x)

    #plot_napfd_metrics(afpd, mean_interval, mean_missed, missed_fit, qax, x)

    if 'comparison' in stats:
        plot_result_difference_bars(stats, 'napfd', rax, x)
        # plot_result_difference_bars(stats,'recall',vax1,x)
    else:
        plot_results_line_graph(stats, 'rewards', mean_interval, rax, x)

    # plot_validation(val_res, lambda res: res['napfd'], 'Validation Results', 'NAPFD', vax1)
    # plot_validation(val_res, lambda res: res['detected'] / (res['detected'] + res['missed']) if (res['detected'] + res['missed']) > 0 else 1,
    #                 'Validation Results', 'Failures Detected (in %)', vax2)

    # if 'comparison' in stats:
    #     plot_result_difference_bars(stats, 'recall', vax1, x)
    #     # plot_result_difference_bars(stats,'recall',vax1,x)
    # else:
    plot_results_line_graph(stats, 'recall', mean_interval, qax2, x)

    if 'comparison' in stats:
        plot_result_difference_bars(stats, 'recall', rax2, x)


    plt.tight_layout()

    if plot_graphs:
        plt.show()

    if save_graphs:
        fig.savefig('_learning.pgf' % prefix, bbox_inches='tight')
        fig.savefig('_learning.png' % prefix, bbox_inches='tight')
        plt.close('all')


def plot_stats_separate_figures(prefix, stats_file, val_file, mean_interval=10, plot_graphs=False, save_graphs=False):
    if not plot_graphs and not save_graphs:
        print('Set at least one of plot_graphs and save_graphs to True')
        return

    sns.set_style('whitegrid')
    sns.set_context('paper')
    stats = pickle.load(open(stats_file, 'rb'))

    failure_count = np.add(stats['detected'], stats['missed'])
    x = range(1, int(len(stats['scenarios']) / mean_interval) + 1)
    perc_missed = [m / fc if fc > 0 else 0 for (m, fc) in zip(stats['missed'], failure_count)]
    mean_missed, missed_fit = mean_values(x, perc_missed, mean_interval)
    mean_reward, reward_fit = mean_values(x, stats['rewards'], mean_interval)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_napfd_metrics([r[3] for r in stats['result']], mean_interval, mean_missed, missed_fit, ax, x)

    if plot_graphs:
        plt.draw()

    if save_graphs:
        fig.savefig('%s_quality.pgf' % prefix, bbox_inches='tight', transparent=True)
        fig.savefig('%s_quality.png' % prefix, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_reward(mean_interval, mean_reward, ax, reward_fit, x)

    if plot_graphs:
        plt.draw()

    if save_graphs:
        fig.savefig('%s_reward.pgf' % prefix, bbox_inches='tight')
        fig.savefig('%s_reward.png' % prefix, bbox_inches='tight')

    val_res = pickle.load(open(val_file, 'rb'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_validation(val_res, lambda res: res['napfd'], 'Validation Results', 'NAPFD', ax)
    if plot_graphs:
        plt.draw()

    if save_graphs:
        fig.savefig('%s_validation_napfd.pgf' % prefix, bbox_inches='tight')
        fig.savefig('%s_validation_napfd.png' % prefix, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_validation(val_res, lambda res: res['detected'] / (res['detected'] + res['missed']) if (res['detected'] + res['missed']) > 0 else 1,
                    'Validation Results', 'Failures Detected (in %)', ax)

    if plot_graphs:
        plt.draw()

    if save_graphs:
        fig.savefig('%s_validation_failures.pgf' % prefix, bbox_inches='tight')
        fig.savefig('%s_validation_failures.png' % prefix, bbox_inches='tight')

    if plot_graphs:
        plt.show()  # Keep window open
    else:
        plt.close('all')


def plot_results_line_graph(stats, metric, mean_interval, qax, x, include_comparison=True):
    if include_comparison and 'comparison' in stats:
        for key in stats['comparison'].keys():
            values, fitline = mean_values(x, stats['comparison'][key][metric], mean_interval)
            if(metric=='napfd'):
                values=values*100
            qax.plot(x, values , label=key)
            #qax.plot(x, fitline(x) * 100, color='black')

    values, fitline = mean_values(x, stats[metric], mean_interval)
    if(metric=='napfd'):
        values=values*100
    qax.plot(x, values , label=metric)
    #qax.plot(x, fitline(x) * 100, color='black')

    qax.set_ylim([0, 100])
    qax.legend(ncol=2)
    qax.set_xlim([1, max(x)])


def plot_napfd_metrics(afpd, mean_interval, mean_missed, missed_fit, qax, x):
    mean_afpd, afpd_fit = mean_values(x, afpd, mean_interval)
    qax.plot(x, mean_afpd * 100, label='NAPFD', color='blue')
    qax.plot(x, afpd_fit(x) * 100, color='black')

    qax.plot(x, mean_missed * 100, label='Percent Missed', color='green')
    qax.plot(x, missed_fit(x) * 100, color='black')
    qax.set_ylim([0, 100])
    qax.legend(ncol=2)
    qax.set_xlim([1, max(x)])
    qax.set_title('Failure Detection (averaged over %d schedules)' % mean_interval)


def plot_reward(mean_interval, mean_reward, rax, reward_fit, x):
    rax.plot(x, mean_reward, label='Reward', color='red')
    rax.plot(x, reward_fit(x), color='black')
    rax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    rax.set_xlim([1, max(x)])
    rax.set_title('Reward (averaged over %d schedules)' % mean_interval)


def plot_validation(val_res, res_fun, title, ylabel, ax=None):
    if not ax:
        ax = plt.gca()

    df = pd.DataFrame.from_dict(val_res)
    # print(type(df))
    # print(len(df))
    # print(df)
    # # print(df[0])
    res_df = df.apply(res_fun, raw=True, axis=1)
    res_df.name = 'res'
    ydat = pd.concat([df, res_df], axis=1)
    # print(ydat)
    # input()
    sns.boxplot(data=ydat, x='step', y='res', palette=sns.color_palette(n_colors=1), ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def mean_values(x, y, mean_interval):
    #mean_y = np.mean(np.array(y).reshape(-1, mean_interval), axis=1)
    mean_y = np.array(y)
    z = np.polyfit(x, mean_y, 6)
    f = np.poly1d(z)
    return mean_y, f


def pickle_to_dataframe(pickle_file):
    return pd.DataFrame.from_dict(pd.read_pickle(pickle_file))


def print_failure_detection(result_dir, file_prefixes):
    df = pd.DataFrame()

    for fp in file_prefixes:
        searchpath = os.path.join(result_dir, fp)
        files = glob.glob(searchpath + '_*_stats.p')

        dfs = pd.concat([pickle_to_dataframe(f) for f in files])
        df = df.append(dfs)

    print (df)

if __name__ == '__main__':
    stats_file = 'tableau_iofrol_timerank_lr0.3_as5_n1000_eps0.1_hist3_tableau_stats.p'
    val_file = 'tableau_iofrol_timerank_lr0.3_as5_n1000_eps0.1_hist3_tableau_val.p'
    mean_interval = 1
    plot_stats_single_figure('tableau', stats_file, val_file, mean_interval, plot_graphs=True, save_graphs=False)
    #plot_stats_separate_figures('netq', stats_file, val_file, mean_interval, plot_graphs=False, save_graphs=True)
    #print_failure_detection('evaluation', ['heur_sort', 'heur_weight', 'random'])


