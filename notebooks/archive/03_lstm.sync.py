#%%
# Routine implemented in peratouch.routines
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import LSTM 
from peratouch.config import path_five_users_main 
from peratouch.data import load_data

input_size = 4 
n_batches = 15
Xraw, yraw = load_data(path_five_users_main)
D = Data(Xraw, yraw)
D.shuffle()
D.make_folds(5)
D.next_fold()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=False)
D.tensors_to_device()
D.print_shapes()
model = LSTM(input_size, hidden_size=4*input_size) 
T = Trainer(D)
T.setup(model, batch_size=int(len(D.xtr)/n_batches), max_epochs=20)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

#%%
# Old routine (has not been updated)
# A notebook for simple lstm exploration
# Case of single lstm cell
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import lstm
from peratouch.config import path_five_users_main 

input_size = 4 
D = Data(path_five_users_main, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=False)
D.tensors_to_device()
D.print_shapes()
model = lstm(input_size, hidden_size=4*input_size) 
T = Trainer(D)
T.setup(model, batch_size=5000, max_epochs=20)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

#%%
# Look at 3 triggers
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import lstm
from peratouch.config import path_five_users_main 

input_size = 32 
D = Data(path_five_users_main, triggers=True, releases=False)
D.group_presses()
D.split()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=False)
D.tensors_to_device()
D.print_shapes()
model = lstm(input_size, hidden_size=int(input_size/2)) 
T = Trainer(D)
T.setup(model, batch_size=5000, max_epochs=20)
T.train_model(model)

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

