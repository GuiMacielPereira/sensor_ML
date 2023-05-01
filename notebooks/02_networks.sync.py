
#%%
from peratouch.routines import run_network
from peratouch.data import load_data
from peratouch.config import path_five_users_main, path_five_users_first
#%%
from peratouch.networks import CNN
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(CNN, Xraw, yraw, n_ch=1, n_epochs=10, n_folds=5, n_runs=1, plots=True, n_batches=20, random_resampling=False)
#%%
from peratouch.networks import LSTM 
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(LSTM, Xraw, yraw, input_size=4, hidden_size=8, n_ch=1, n_epochs=10, n_folds=5, n_runs=1, plots=True, n_batches=20, random_resampling=False)
#%%
from peratouch.networks import CNN_LSTM 
Xraw, yraw = load_data(path_five_users_main)
_ = run_network(CNN_LSTM, Xraw, yraw, n_ch=1, n_epochs=10, n_folds=5, n_runs=1, plots=True, n_batches=20, random_resampling=False)
