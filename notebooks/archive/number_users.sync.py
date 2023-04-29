#%%
from peratouch.data import Data, load_data
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
from peratouch.config import path_five_users_main, path_five_users_first
import sklearn
import numpy as np

def run_n_users(X, y, n_folds=5):
    """
    Runs entire routine of fitting CNN model to dataset (X, y)self.
    Performs Cross-Validation of n_folds on input dataset.
    Assumes data is already shuffled.
    """

    D = Data(X, y)

    n_out = len(np.unique(y))

    # Create indices of several folds
    D.make_folds(n_folds)     # Makes indices available inside class

    predictions = []
    actual_vals = []

    # for _ in range(n_folds):     # Run all folds 
    for _ in range(1):     # Run all folds
        D.next_fold()
        D.normalize()
        D.tensors_to_device()
        D.print_shapes()
        # D.plot_data()
        model = CNN(n_ch=1, out_size=n_out)      # Initialize new model each fold
        T = Trainer(D)
        T.setup(model, max_epochs=20, batch_size=int(len(D.xtr)/20))       # 20 minibatches
        T.train_model(model)
        # T.plot_train()
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

results_dict = {}

number_users = range(3, 4)
for n_users in number_users:     # Number of possible users: 2, 3, 4, 5

    user_combinations = itertools.combinations(range(5), n_users)      # From 5 users choose n_users

    for users in user_combinations:
        print("\n\nRunning combination of users ", users)

        Xraw, yraw = load_data(path_five_users_main)

        # Choose one given combination of users
        Xraw = np.concatenate([Xraw[yraw==u] for u in users])
        yraw = np.concatenate([yraw[yraw==u] for u in users])

        # Change labels to be fit increasing range ie. 0, 1, 2, ....
        for i, u in enumerate(users):
            yraw[yraw==u] = i

        # Shuffle data to destroy ordering of users
        Xraw, yraw = sklearn.utils.shuffle(Xraw, yraw, random_state=42)

        # Run same routine for users selected
        results_dict[str(users)] = run_n_users(Xraw, yraw)

        # Store results at each run
        np.savez(str(path_analysis_results / "number_users.npz"), **results_dict)

#%%
stored_results = np.load(str(path_analysis_results / "number_users.npz"))

dict_plot = {}
# Initialize lists for numner of users
for n in number_users:
    dict_plot[n] = []

for key in stored_results:
    print(key, " : ", len(results_dict[key]))
    print(sklearn.metrics.classification_report(*results_dict[key]))
    actual, pred = stored_results[key]
    dict_plot[len(eval(key))].append(np.mean(actual==pred))

#%%
for key in dict_plot:
    run_data = dict_plot[key]
    print(key, " : ", run_data)
    print(key, ": ", np.mean(run_data), " +/- ", np.std(run_data))





