#%%
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# %cd gdrive/MyDrive/Masters_Project
# %pip install -e peratouch

#%%
from peratouch.data import Data, load_data
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
from peratouch.config import path_five_users_main, path_five_users_first
import sklearn
import numpy as np
from datetime import date

def run_n_presses_users(X, y, n_press, n_folds=5, plotting=True):
    """
    Runs entire routine of fitting CNN model to dataset (X, y)self.
    Performs Cross-Validation of n_folds on input dataset.
    Assumes data is in temporal order, and NOT shuffled.
    """

    D = Data(X, y)

    n_out = len(np.unique(y))

    D.group_presses(n_press=n_press)
    D.shuffle()

    # Create indices of several folds
    D.make_folds(n_folds)     # Makes indices available inside class

    predictions = []
    actual_vals = []

    for i in range(1):     # Run all folds
        D.next_fold()
        # D.resample_presses(n_press)
        D.normalize()
        D.tensors_to_device()
        D.print_shapes()
        model = CNN(n_ch=n_press, out_size=n_out)      # Initialize new model each fold
        T = Trainer(D)
        T.setup(model, max_epochs=30, batch_size=int(len(D.xtr)/20))       # 20 minibatches
        T.train_model(model)
        if (plotting & (i==0)):
            T.plot_train()
        R = Results(D, model)
        R.test_metrics()
        preds, actual = R.get_preds_actual()

        predictions.extend(preds)
        actual_vals.extend(actual)

    print(sklearn.metrics.classification_report(actual_vals, predictions))
    return actual_vals, predictions

#%%
from peratouch.config import path_analysis_results
import itertools

number_presses = [1, 3, 5, 10, 20]
number_users = range(2, 6)

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

