# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import CNN
# from peratouch.config import datapath_five_users
from peratouch.config import path_five_users_main, path_five_users_first

D = Data(path_five_users_main, triggers=True, releases=False)
D.shuffle()
D.halve_raw_data()

# Create indices of several folds
n_folds = 5
D.make_folds(n_folds)     # Makes indices available inside class

predictions = []
actual_vals = []

n_runs = 1
for f in range(n_runs):
    D.next_fold()
    D.normalize()
    D.tensors_to_device()
    D.print_shapes()
    # D.plot_data()
    model = CNN(n_ch=1)      # Initialize new model each fold
    T = Trainer(D)
    T.setup(model, max_epochs=20, batch_size=int(len(D.xtr)/20))       # 20 minibatches
    T.train_model(model)
    T.plot_train()
    R = Results(D, model)
    preds, actual = R.get_preds_actual()
    # R.test_metrics(report=True, conf_matrix=False)
    # R.find_most_uncertain_preds()

    predictions.extend(preds)
    actual_vals.extend(actual)

import sklearn
print(sklearn.metrics.classification_report(actual_vals, predictions))
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

