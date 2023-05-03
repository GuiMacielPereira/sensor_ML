import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from cycler import cycler
from peratouch.config import path_analysis_figures, path_figures

sns.set_theme()

# Plot grid of triggers
def plot_grid(batch):
    batch = batch.reshape(batch.shape[0], -1)
    if len(batch) > 50:   # Cut batch short if required
        batch = batch[:30]

    nx_plots = 10
    ny_plots = int(np.ceil(len(batch) / nx_plots))
    
    plt.figure(figsize=(nx_plots*2, ny_plots*2))
    plt.tight_layout()
    for i, sig in enumerate(batch):
        plt.subplot(ny_plots, nx_plots, i+1)
        plt.plot(range(len(sig)), sig, "b.")
        plt.xticks([])
        # plt.ylim(0, batch.max())
        if i%nx_plots: plt.yticks([])


# Concatenate triggers and plot continuously
def plot_flatten(batch):
    points = batch.flatten()
    if points.size==0: return 
    plt.figure(figsize=(30, 5))
    plt.tight_layout()
    plt.plot(range(len(points)), points, "b.")


# Plot input data X, i.e. user profiles 
def plot_X(X, y):
    """Plots mean and std of user triggers"""
    _, n_ch, n_points = X.shape
    n_users = len(np.unique(y))
    plt.figure(figsize=(n_users*3, n_ch*3))
    plt.suptitle("Mean and std of signals for users and channels")
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
def plot_trainer(epochs, losses, accuracies, model_name, plot_loss=True, plot_acc=True):
    """ Plot accuracies and losses during training of the model """

    # First, set cycler for colors and linestyles
    colors = sns.color_palette("husl", 9)
    # Build repeating colors and linestyles
    colors = [(c, c) for c in colors]
    lines = [('--', '-') for c in colors]
    # Flatten list of sublists
    colors = [item for sublist in colors for item in sublist]
    lines = [item for sublist in lines for item in sublist]

    # plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', lines))):
    with mpl.rc_context({'axes.prop_cycle' : f'(cycler(color={colors}) + cycler(linestyle={lines}))'}):

        # marker = 'D'
        plt.figure()

        if plot_loss:
            plt.plot(epochs, losses, label=[f"{model_name} Train Loss", f"{model_name} Val Loss"])
        if plot_acc:
            plt.plot(epochs, accuracies, label=[f"{model_name} Train Acc", f"{model_name} Val Acc"])

    plt.ylim(top=1)
    plt.xlim(left=1)
    plt.legend()
    plt.xlabel("Epochs")
    # plt.xticks(epochs)

    filename = model_name + '_training.pdf'
    plt.savefig(str(path_figures / filename), bbox_inches='tight')

# Plots for analysis 
# Dataset sizes
def plot_dataset_sizes(load_path, xlabel='Train dataset size'):

    stored_results = np.load(load_path) 

    x = []
    y = []

    for key in stored_results:
        x.append(int(key))
        actual_vals, preds = stored_results[key]
        y.append(np.mean(np.array(actual_vals)==np.array(preds)))

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'k-x', label="n_users=5, n_presses=1", markersize=7)
    plt.ylabel('Accuracy test')
    plt.xlabel(xlabel)
    plt.legend()
    plt.xscale('log')
    plt.xticks(x)
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

    plt.figure(figsize=(8,5))

    for n_p in n_presses:
        x = [int(key.split('_')[0]) for key in user_groups if key.split('_')[-1]==n_p]
        y = [np.mean(user_groups[key]) for key in user_groups if key.split('_')[-1]==n_p]
        e = [np.std(user_groups[key]) for key in user_groups if key.split('_')[-1]==n_p]
        plt.errorbar(x, y, e, label=f'n_presses={n_p}', 
                fmt='--D', capsize=4, markersize=5, linewidth=1)

    plt.xlabel('Number of users')
    plt.ylabel('Acccuracy test')
    plt.xticks([2, 3, 4, 5])
    plt.legend()

    filename = str(load_path).split('/')[-1].split('.')[0] + '.pdf'
    plt.savefig(str(path_analysis_figures / filename), bbox_inches='tight')
