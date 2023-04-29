import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

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


def plot_flatten(batch):
    points = batch.flatten()
    if points.size==0: return 
    plt.figure(figsize=(30, 5))
    plt.tight_layout()
    plt.plot(range(len(points)), points, "b.")


def plot_X(X, y):
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

        x += n_points 
    
# Plots for analysis
import sklearn

def plot_dataset_sizes(save_path, xlabel='Train dataset size'):

    stored_results = np.load(save_path) 

    x = []
    y = []

    for key in stored_results:
        x.append(int(key))
        actual_vals, preds = stored_results[key]
        y.append(np.mean(np.array(actual_vals)==np.array(preds)))

    plt.figure()
    plt.plot(x, y, 'k-x')
    plt.ylabel('Accuracy')
    plt.xlabel(xlabel)
    plt.ylim(0,1)
    

def plot_presses_users(save_path):

    stored_results = np.load(save_path) 

    user_groups = {}

    for key in stored_results:
        users, n_p = key.split('_')

        new_key  = f'{len(eval(users))}_{n_p}'

        if new_key not in user_groups:
            user_groups[new_key] = []

        act_vals, preds = stored_results[key]
        user_groups[new_key].append(np.mean(act_vals==preds))

    for k in user_groups:
        print(f'{k} : {user_groups[k]}')
    

