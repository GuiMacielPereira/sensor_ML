
#%%
from peratouch.data import Data, load_data
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
from peratouch.config import path_five_users_main, path_five_users_first
import sklearn

def run_dataset(X, y):
    """
    Runs entire routine of fitting CNN model to dataset (X, y)self.
    Performs Cross-Validation of n_folds.
    Assumes data is already shuffled.
    """

    D = Data(X, y)

    # Create indices of several folds
    n_folds = 5               # Run 5 folds for each dataset
    D.make_folds(n_folds)     # Makes indices available inside class

    predictions = []
    actual_vals = []

    for _ in range(n_folds):     # Run all folds 
        D.next_fold()
        D.normalize()
        D.tensors_to_device()
        D.print_shapes()
        # D.plot_data()
        model = CNN(n_ch=1)      # Initialize new model each fold
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
# Run CNN routine for datasets with different sizes
# Goal is to analyse how dataset size affects performance

# The routine below is a dirty trick using the capabilities of k-fold from sklearn
# Takes test set from k-fold and uses it to subsample the raw dataset
# Used it like this because it automates the majority of the work

Xraw, yraw = load_data(path_five_users_main)
# Shuffle data to destroy ordering of users
Xraw, yraw = sklearn.utils.shuffle(Xraw, yraw, random_state=42)

results_dict = {}

for n_splits in range(2, 4):         # Splits of raw dataset
    print("\n\n--- Testing new dataset size ---\n\n")
    kf = sklearn.model_selection.KFold(n_splits)       # No shuffling

    actual_vals, predictions = [], []

    for (_, data_idx) in kf.split(Xraw):
        print("\n-- New splitting of dataset --\n")

        X = Xraw[data_idx]
        y = yraw[data_idx]
         
        actual, preds = run_dataset(X, y)

        actual_vals.extend(actual)
        predictions.extend(preds)

    # TODO: Change to record len of training dataset, not total dataset
    results_dict[str(len(X))] = (actual_vals, predictions)
    

#%%
print("len of raw data: ", len(Xraw))
for key in results_dict:
    print(key, " : ", len(results_dict[key][0]))
    print(sklearn.metrics.classification_report(*results_dict[key]))
