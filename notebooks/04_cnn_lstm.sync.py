
#%%
# Look at stardard cnn_lstm for 3 signals
from peratouch.data import Data 
from peratouch.trainer import Trainer
from peratouch.results import Results 
from peratouch.networks import cnn_lstm
from peratouch.config import datapath_five_users

D = Data(datapath_five_users, triggers=True, releases=False)
D.group_presses()
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
model = cnn_lstm(input_ch=3, n_filters_start=8, hidden_lstm=16, out_size=5) 
T = Trainer(D)
T.setup(model, learning_rate=5e-2, weight_decay=1e-3, batch_size=5000, max_epochs=20, verbose=True)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()


#%%
# Case of time-distributed cnn-lstm 
from peratouch.data import Data 
from peratouch.trainer import Trainer
from peratouch.results import Results 
from peratouch.networks import cnn_lstm_time_distributed
from peratouch.config import datapath_five_users

input_size = 32 
D = Data(datapath_five_users, triggers=True, releases=False)
D.group_presses()
D.split()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=False)
D.tensors_to_device()
D.print_shapes()
model = cnn_lstm_time_distributed(input_size=input_size, out_size=5, global_pool=False) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=5000, max_epochs=20)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()
