#%%
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# %cd gdrive/MyDrive/Masters_Project
# %pip install -e peratouch

#%%
from peratouch.data import load_data
from peratouch.routines import run_network
from peratouch.config import path_five_users_main, path_five_users_first
from datetime import date
from peratouch.networks import CNN
import numpy as np
import sklearn

def run_n_presses_users(X, y, n_press):
  return run_network(CNN, X, y, n_ch=n_press, n_epochs=100, n_folds=5, n_runs=5, plots=False, n_batches=15, random_resampling=False)

#%%
from peratouch.config import path_analysis_results
import itertools

number_presses = [1, 3, 5, 10, 20]
number_users = range(3, 6)

results_dict = {}

Xraw, yraw = load_data(path_five_users_main)

for n_users in number_users:     # Number of possible users: 2, 3, 4, 5

    user_combinations = itertools.combinations(range(5), n_users)      # From 5 users choose n_users

    for users in user_combinations:
        print("\n\nRunning combination of users ", users)

        # Choose one given combination of users
        X = np.concatenate([Xraw[yraw==u] for u in users])
        y = np.concatenate([yraw[yraw==u] for u in users])

        # Change labels to be fit increasing range ie. 0, 1, 2, ....
        for i, u in enumerate(users):
            y[y==u] = i

        for n_press in number_presses:

            results_dict[f'{users}_{n_press}'] = run_n_presses_users(X, y, n_press)

            np.savez(str(path_analysis_results / f"no_presses_users_{date.today()}.npz"), **results_dict)

#%%
stored_results = np.load(str(path_analysis_results / f"no_presses_users_{date.today()}.npz"))

for key in stored_results:
    print(key, ":\n", sklearn.metrics.classification_report(*results_dict[key]))

