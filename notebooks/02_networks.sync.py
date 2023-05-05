#%%
# Uncomment if running from Google Colab
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# %cd gdrive/MyDrive/Masters_Project
# %pip install -e peratouch

#%%
from peratouch.routines import run_network
from peratouch.data import load_data
from peratouch.config import path_five_users_main, path_five_users_first

#%%
from peratouch.networks import CNN
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(CNN, Xraw, yraw, n_ch=1, n_epochs=100, n_folds=10, n_runs=10, plots=False, n_batches=15, random_resampling=False)

#%%
from peratouch.networks import LSTM 
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(LSTM, Xraw, yraw, input_size=16, hidden_size=16, n_ch=1, n_epochs=100, n_folds=10, n_runs=10, plots=False, n_batches=15, random_resampling=False)

#%%
from peratouch.networks import CNN_LSTM 
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(CNN_LSTM, Xraw, yraw, n_ch=1, n_epochs=100, n_folds=10, n_runs=10, plots=False, n_batches=15, random_resampling=False)

#%%
# # Ran this routine only once for a simple comparison
# ## Run comparison of input sizes of LSTM
# from peratouch.config import path_analysis_results
# from datetime import date
# from peratouch.networks import LSTM 
# import numpy as np

# Xraw, yraw = load_data(path_five_users_main)

# input_sizes = [1, 2, 4, 8, 16, 32]
# results = {}
# for in_size in input_sizes:
#     print('----- input size = ', in_size, "\n")
#     results[str(in_size)] = run_network(LSTM, Xraw, yraw, input_size=in_size, hidden_size=16, n_ch=1, n_epochs=50, n_folds=5, n_runs=5, plots=False, n_batches=15, random_resampling=False)

# save_path = str(path_analysis_results/f'input_size_lstm_{date.today()}.npz')
# np.savez(save_path, **results)

# data = np.load(save_path)
# x = [int(k) for k in data]
# y = [np.mean(vals==preds) for (vals, preds) in data.values()]
# import matplotlib.pyplot as plt 
# plt.plot(x, y)

