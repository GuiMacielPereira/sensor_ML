#%%
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

