from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# change font size
matplotlib.rcParams.update({'font.size': 13})


def plot_median_mean(prob_t, prob_l, title=None, y_label='Probability', scale='both', type='median', save_path=None):
    # Create figure based on the scale option
    if scale == 'both':
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = None

    # Function to plot curves
    def plot_curves(ax, scale, title=None):

        if type=='mean':

            mean_t = prob_t.mean(axis=1)
            std_t = prob_t.std(axis=1)
            mean_l = prob_l.mean(axis=1)
            std_l = prob_l.std(axis=1)

            ax.plot(mean_t, color='tab:blue', label='truth mean', linestyle='-')
            ax.fill_between(range(len(mean_t)), mean_t - std_t, mean_t + std_t, color='tab:blue', alpha=0.1)

            ax.plot(mean_l, color='tab:orange', label='lie mean', linestyle='-')
            ax.fill_between(range(len(mean_l)), mean_l - std_l, mean_l + std_l, color='tab:orange', alpha=0.1)

        elif type=='median':
            median_t = prob_t.median(axis=1).values
            median_l = prob_l.median(axis=1).values
            quantile_25_t = prob_t.quantile(0.25, axis=1)
            quantile_75_t = prob_t.quantile(0.75, axis=1)
            quantile_25_l = prob_l.quantile(0.25, axis=1)
            quantile_75_l = prob_l.quantile(0.75, axis=1)

            ax.plot(median_t, color='tab:blue', label='truth median', linestyle='-')
            ax.fill_between(range(len(median_t)), quantile_25_t, quantile_75_t, color='tab:blue', alpha=0.1)

            ax.plot(median_l, color='tab:orange', label='lie median', linestyle='-')
            ax.fill_between(range(len(median_l)), quantile_25_l, quantile_75_l, color='tab:orange', alpha=0.1)

        #ax.plot(prob_t.median(axis=1).values, color='tab:blue', label='truth median')
        #ax.plot(prob_l.median(axis=1).values, color='tab:orange', label='lie median')


        ax.grid()
        ax.set_xlabel("layer_id")
        ax.set_ylabel(y_label)
        title = '' if not title else title + f' ({scale} scale)'
        ax.set_title(title)
        ax.legend(loc='best')

    # Plot linear scale
    if scale == 'linear':
        plot_curves(ax1, 'linear', title)
    elif scale == 'log':
        plot_curves(ax1, 'log', title)
        ax1.set_yscale('log')
    else:
        plot_curves(ax1, 'linear', title)
        plot_curves(ax2, 'log', title)
        ax2.set_yscale('log')

    if save_path:
        fig.savefig(save_path)

    plt.show()



def plot_h_bar(prob_truth, prob_lie, selected_layers, title=None, y_label="top tokens", save_path=None):
    plt.rc('font', size=13)
    width = 0.5
    k = prob_truth.shape[0]
    fig, axs = plt.subplots(1, len(selected_layers), figsize=(len(selected_layers)*2.5, 5))

    prob_truth_means, prob_truth_medians = prob_truth.mean(dim=-1), prob_truth.median(dim=-1).values
    prob_lie_means, prob_lie_medians = prob_lie.mean(dim=-1), prob_lie.median(dim=-1).values

    for i, l in enumerate(selected_layers):
        y = np.arange(k)
        axs[i].barh(y - width/2, prob_truth_medians[:, l], height=width/3, color='tab:blue', align='center', label='truth median', edgecolor='black')
        axs[i].barh(y - width/4, prob_truth_means[:, l], height=width/3, color='tab:blue', align='center', label='truth mean',hatch='//', edgecolor='black')
        axs[i].barh(y + width/4, prob_lie_medians[:, l], height=width/3, color='tab:orange', align='center', label='lie median', edgecolor='black')
        axs[i].barh(y + width/2, prob_lie_means[:, l], height=width/3, color='tab:orange', align='center', label='lie mean', hatch = '//', edgecolor='black')
        axs[i].grid('off')
        axs[i].set_yticks(np.arange(k))
        axs[i].set_yticklabels([])
        if i == 0:
            axs[i].set_ylabel(y_label)
            axs[i].set_yticklabels(np.arange(1, k+1).astype(int))
        if i ==  len(selected_layers)-1:
            axs[i].legend(loc='best')
        axs[i].set_xlabel(f'\nlayer_id: {l}')

    fig.align_labels()
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path)
    plt.show()
