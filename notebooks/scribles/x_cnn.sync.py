# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
# Standard example 
from peratouch.data import Data 
from peratouch.trainer import Trainer 
from peratouch.results import Results 
from peratouch.networks import X_CNN
from peratouch.config import path_five_users_first 

D = Data(path_five_users_first, triggers=True, releases=False)
D.split()
# D.balance_train()
D.normalize()
D.tensors_to_device()
D.print_shapes()
D.plot_data()
#%%
model = X_CNN(input_ch=1, n_filters=8, n_hidden=128, out_size=5)
T = Trainer(D)
T.setup(model, max_epochs=20, batch_size=5000)
T.train_model(model)
T.plot_train()

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

#%%
from peratouch.plot import plot_grid, plot_flatten
import matplotlib.pyplot as plt
# Look at first layer
input = D.xte[500:501]
plt.plot(range(len(input.squeeze())), input.squeeze(), 'b.')

out = model.first_layer(input)
plot_grid(out.data.squeeze())
out = model.second_layer(out)
plot_grid(out.data.squeeze())
out = model.third_layer(out)
plot_grid(out.data.squeeze())

#%%
# Look at 3 channels
from peratouch.data import Data 
from peratouch.trainer import Trainer  
from peratouch.networks import X_CNN 
from peratouch.config import datapath_five_users
D = Data(datapath_five_users, triggers=True, releases=False)
D.group_presses()
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
# Did not see any improvement by trying out CNN_Dense
model = X_CNN(input_ch=3, n_filters=16, n_hidden=256, out_size=5) 
T = Trainer(D) 
T.setup(model,learning_rate=1e-2, weight_decay=1e-3, max_epochs=50, batch_size=700)
T.train_model(model)

R = Results(D, model)
R.test_metrics(report=True, conf_matrix=True)
R.find_most_uncertain_preds()

#%%
from peratouch.plot import plot_grid, plot_flatten
import matplotlib.pyplot as plt
# Look at first layer
input = D.xte[500:501]
plot_flatten(input)

out = model.first_layer(input)
plot_grid(out.data.squeeze())
out = model.second_layer(out)
plot_grid(out.data.squeeze())
out = model.third_layer(out)
plot_grid(out.data.squeeze())
