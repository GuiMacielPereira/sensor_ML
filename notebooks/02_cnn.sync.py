# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
# Standard example 
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
# from peratouch.config import datapath_five_users
from peratouch.config import path_five_users_main, path_five_users_first

D = Data(path_five_users_main, triggers=True, releases=False)
D.shuffle()
D.halve_raw_data()
D.make_folds()
for f in range(5):
    print("\n\n-- Next Fold --")
    D.next_fold()
    # D.split()
    D.normalize()
    D.tensors_to_device()
    D.print_shapes()
    D.plot_data()
#%%
model = CNN(input_ch=1)
T = Trainer(D)
T.setup(model, max_epochs=20, batch_size=5000)
T.train_model(model)
# T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=False)
R.find_most_uncertain_preds()
#%%
# Look at 3 channels
from peratouch.data import Data 
from peratouch.trainer import Trainer  
from peratouch.networks import CNN 
from peratouch.config import path_five_users_main, path_five_users_first
from peratouch.results import Results
D = Data(path_five_users_main, triggers=True, releases=False)
D.group_presses()
D.split()
D.normalize()
# D.resample_triggers()
D.tensors_to_device()
D.print_shapes()
# Did not see any improvement by trying out CNN_Dense
model = CNN(input_ch=3) 
T = Trainer(D) 
T.setup(model,learning_rate=1e-2, weight_decay=1e-3, max_epochs=20, batch_size=5000)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

