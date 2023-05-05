#%%
# # Uncomment if running from Google Colab
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# %cd gdrive/MyDrive/Masters_Project
# %pip install -e peratouch
#%%
# Run CNN routine for datasets with different sizes
# Goal is to analyse how dataset size affects performance
from peratouch.data import load_data
from peratouch.config import path_five_users_main, path_five_users_first
from peratouch.config import path_analysis_results
from peratouch.routines import run_network
from peratouch.networks import CNN
from datetime import date
import numpy as np
import sklearn

n_folds = 5

def run_dataset(X, y):
    return run_network(CNN, X, y, n_ch=1, n_epochs=100, n_folds=n_folds, n_runs=n_folds, plots=False, n_batches=15, random_resampling=False)

Xraw, yraw = load_data(path_five_users_main)
Xraw, yraw = sklearn.utils.shuffle(Xraw, yraw)

results_dict = {}

for n_splits in [80, 40, 20, 10, 5, 4, 3, 2, 1]:         # Splits of raw dataset
    print("\n\n--- Testing new dataset size ---\n\n")

    actual_vals, predictions = [], []

    if n_splits==1:    # Case of entire dataset
        X = Xraw
        y = yraw

        actual, preds = run_dataset(X, y)

        actual_vals.extend(actual)
        predictions.extend(preds)

    else:    
      kf = sklearn.model_selection.KFold(n_splits)       # No shuffling
      for i, (_, data_idx) in enumerate(kf.split(Xraw)):
          print("\n-- New splitting of dataset --\n")

          X = Xraw[data_idx]
          y = yraw[data_idx]
          
          actual, preds = run_dataset(X, y)

          actual_vals.extend(actual)
          predictions.extend(preds)

    # NOTE: Hardcoded value of 0.85 below is the split between training and validation sets
    tr_size = int(len(X)*(n_folds-1)/n_folds * 0.85)         # Training size is size of all folds except one 
    results_dict[str(tr_size)] = (actual_vals, predictions)
    filename = str(path_analysis_results / f"dataset_size_{date.today()}.npz") 
    np.savez(filename, **results_dict)

#%%
# Loads data from stored dict
stored_results = np.load(filename)
print("len of raw data: ", len(Xraw))
for key in stored_results:
    print(key, " : ", len(results_dict[key][0]))
    print(sklearn.metrics.classification_report(*results_dict[key]))

