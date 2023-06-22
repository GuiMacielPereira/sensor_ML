import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from cycler import cycler
import itertools
from peratouch.config import path_analysis_figures, path_figures

sns.set_theme()
# sns.set_context('talk')
# sns.set_palette('husl')

# Plot grid of triggers
def plot_grid(batch):
    batch = batch.reshape(batch.shape[0], -1)
    if len(batch) > 50:   # Cut batch short if required
        batch = batch[:36]

    nx_plots = 6 
    ny_plots = int(np.ceil(len(batch) / nx_plots))
    
    plt.figure(figsize=(nx_plots*2, ny_plots*2))
    plt.tight_layout()
    for i, sig in enumerate(batch):
        plt.subplot(ny_plots, nx_plots, i+1)
        plt.ylim(-0.3, 1.03*batch.max())
        plt.plot(range(len(sig)), sig, "b.")
        plt.xticks([])
        if i%nx_plots: 
            plt.yticks([])
        else:
            plt.ylabel('Voltage [V]')
        plt.grid(False)

    filename = 'triggers_grid.pdf'
    plt.savefig(str(path_figures / filename), bbox_inches='tight')

# Concatenate triggers and plot continuously
def plot_flatten(batch):
    points = batch.flatten()
    if points.size==0: return 
    plt.figure(figsize=(15, 3))
    plt.tight_layout()
    plt.plot(range(len(points)), points, "b.")

    plt.ylabel('Voltage[V]')
    plt.xlabel('Number of points')

    # filename = 'flat_signal.pdf'
    # plt.savefig(str(path_figures / filename), bbox_inches='tight')

# Plot input data X, i.e. user profiles 
def plot_X(X, y):
    """Plots mean and std of user triggers"""
    _, n_ch, n_points = X.shape
    n_users = len(np.unique(y))
    plt.figure(figsize=(n_users*3, n_ch*3))
    plt.tight_layout()

    x = np.arange(n_points)
    for u in np.unique(y):
        Xuser = X[y==u]
        Xmean = Xuser.mean(axis=0, keepdims=False)
        Xstd = Xuser.std(axis=0, keepdims=False)

        for j, (mean, std) in enumerate(zip(Xmean, Xstd)):
            plt.subplot(n_ch, 1, j+1)
            plt.plot(x, mean, marker='.')
            plt.fill_between(x, mean-std, mean+std, alpha=0.2)
            plt.xticks([])
        plt.ylabel("Voltage [V]")

        x += n_points 

    filename = 'mean_std_users.pdf'
    plt.savefig(str(path_figures / filename), bbox_inches='tight')
    
# Plot for training
def plot_trainer(epochs, losses, accuracies, model_name, save_path):
    """ Plot accuracies and losses during training of the model """

    colors = ['limegreen', 'darkgreen', 'deepskyblue', 'darkblue']

    with sns.axes_style('dark'):
        with mpl.rc_context({'axes.prop_cycle' : f'(cycler(color={colors}))'}):

            # marker = 'D'
            fig, ax0 = plt.subplots()
            ax1 = ax0.twinx()

            ax1.plot(epochs, losses, label=[f"{model_name} Train Loss", f"{model_name} Val Loss"])
            next(ax0._get_lines.prop_cycler)
            next(ax0._get_lines.prop_cycler)
            ax0.plot(epochs, accuracies, label=[f"{model_name} Train Acc", f"{model_name} Val Acc"])
            # ax0.set_ylim(top=1)

            ax0.legend(bbox_to_anchor=(0.45, 1.17))
            ax1.legend(bbox_to_anchor=(0.97, 1.17))

            ax0.set_ylabel('Accuracy', color="blue")
            ax0.tick_params(axis='y', colors='blue')
            ax0.set_ylim(bottom=0.7*np.min(accuracies))

            ax1.set_ylabel('Loss', color="green")
            ax1.tick_params(axis='y', colors='green')
            ax1.set_ylim(top=1.4*np.max(losses))

            ax0.set_xlabel('Epochs')
            ax0.set_xlim(left=1)

    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight')

# Plots for analysis 

# LSTM architecture
def plot_lstm_sizes(load_path):
    stored_results = np.load(load_path) 

    acc_results = {}
    for k in stored_results:
        v, p = stored_results[k]
        acc_results[k] = np.mean(v==p)

    plt.figure(figsize=(7,6))

    markers = itertools.cycle(('^', 'o', 's', 'D'))

    for hid_size in ['8', '16', '32']:
        x = [int(key.split('_')[0]) for key in stored_results if key.split('_')[-1]==hid_size]
        y = [acc_results[key] for key in stored_results if key.split('_')[-1]==hid_size]
        plt.plot(x, y, ':',  label=f'LSTM hidden size={hid_size}', 
                marker=next(markers), markersize=7, linewidth=2)

    plt.xlabel('Input size of LSTM cell')
    plt.ylabel('Test Acccuracy')
    plt.xscale('log')
    plt.xticks([1, 2, 4, 8, 16, 32])
    plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.legend()

    filename = str(load_path).split('/')[-1].split('.')[0] + '.pdf'
    plt.savefig(str(path_analysis_figures / filename), bbox_inches='tight')

# Dataset sizes
def plot_dataset_sizes(load_path, xlabel='Train dataset size'):

    stored_results = np.load(load_path) 

    x = []
    y = []

    for key in stored_results:
        x.append(int(key))
        actual_vals, preds = stored_results[key]
        y.append(np.mean(np.array(actual_vals)==np.array(preds)))

    plt.figure(figsize=(7, 6))
    plt.plot(x, y, 'k-x', label="n_users=5, n_presses=1", markersize=10)
    plt.ylabel('Test Accuracy')
    plt.xlabel(xlabel)
    plt.legend()
    # plt.xscale('log')
    xticks = x.pop(1)
    plt.xticks(x, rotation=45)
    plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.ylim(0.62,0.72)

    filename = str(load_path).split('/')[-1].split('.')[0] + '.pdf'
    plt.savefig(str(path_analysis_figures / filename), bbox_inches='tight')
    

# Number of presses AND number of users
def plot_presses_users(load_path):
    stored_results = np.load(load_path) 

    user_groups = {}

    for key in stored_results:
        users, n_p = key.split('_')

        new_key  = f'{len(eval(users))}_{n_p}'

        if new_key not in user_groups:
            user_groups[new_key] = []

        act_vals, preds = stored_results[key]
        user_groups[new_key].append(np.mean(act_vals==preds))

    # Find range of number of presses
    n_presses = np.unique([key.split('_')[-1] for key in user_groups])

    # Sort by increasing press
    n_presses = [int(s) for s in n_presses]    # Pass to ints
    n_presses.sort() 
    n_presses = [str(i) for i in n_presses]    # Pass to stings again

    plt.figure(figsize=(7,6))

    for n_p in n_presses:
        x = [int(key.split('_')[0]) for key in user_groups if key.split('_')[-1]==n_p]
        y = [np.mean(user_groups[key]) for key in user_groups if key.split('_')[-1]==n_p]
        e = [np.std(user_groups[key]) for key in user_groups if key.split('_')[-1]==n_p]
        plt.errorbar(x, y, e, label=f'n_presses={n_p}', 
                fmt='--D', capsize=4, markersize=6, linewidth=1.5)

    plt.xlabel('Number of users')
    plt.ylabel('Test Acccuracy')
    plt.xticks([2, 3, 4, 5])
    plt.legend()

    filename = str(load_path).split('/')[-1].split('.')[0] + '.pdf'
    plt.savefig(str(path_analysis_figures / filename), bbox_inches='tight')
