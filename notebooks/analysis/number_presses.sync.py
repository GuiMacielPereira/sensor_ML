#%%
from peratouch.data import Data, load_data
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
from peratouch.config import path_five_users_main, path_five_users_first
import sklearn
import numpy as np

def run_n_presses(X, y, n_press, n_folds=5):
    """
    Runs entire routine of fitting CNN model to dataset (X, y)self.
    Performs Cross-Validation of n_folds on input dataset.
    Assumes data is in temporal order, and NOT shuffled.
    """

    D = Data(X, y)
    D.group_presses(n_press=n_press)
    D.shuffle()

    # Create indices of several folds
    D.make_folds(n_folds)     # Makes indices available inside class

    predictions = []
    actual_vals = []

    # for _ in range(n_folds):     # Run all folds 
    for _ in range(1):     # Run all folds
        D.next_fold()
        # D.resample_presses(n_press)
        D.normalize()
        D.tensors_to_device()
        D.print_shapes()
        # D.plot_data()
        model = CNN(n_ch=n_press, out_size=5)      # Initialize new model each fold
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

Xraw, yraw = load_data(path_five_users_main)

number_presses = [20] # [1, 3, 5, 10, 20]

results_dict = {}
for n_press in number_presses:

    results_dict[str(n_press)] = run_n_presses(Xraw, yraw, n_press)

    np.savez(str(path_analysis_results / "number_presses.npz"), **results_dict)

#%%
stored_results = np.load(str(path_analysis_results / "number_presses.npz"))

for key in stored_results:
    print(key, ":\n", sklearn.metrics.classification_report(*results_dict[key]))
