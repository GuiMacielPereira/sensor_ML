# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
# Standard example 
from peratouch.data import Data, test_accuracy
from peratouch.trainer import Trainer, plot_train 
from peratouch.networks import CNN
from peratouch.config import datapath_five_users

D = Data(datapath_five_users, triggers=True, releases=False)
D.shuffle()
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
models = [CNN(input_ch=1, n_filters=8, n_hidden=128, out_size=5)]
trainers = [Trainer(D)]
for model, T in zip(models, trainers):
    T.setup(model, max_epochs=50, batch_size=5000)
    T.train_model(model)
plot_train(trainers)
test_accuracy([D, D], models)

#%%
# Look at 3 channels
from peratouch.data import Data, test_accuracy
from peratouch.trainer import Trainer , plot_train
from peratouch.networks import CNN 
from peratouch.config import datapath_five_users
D = Data(datapath_five_users, triggers=True, releases=False)
D.group_presses()
D.split()
D.shuffle_presses_train()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
# Did not see any improvement by trying out CNN_Dense
model = CNN(input_ch=3, n_filters=16, n_hidden=256, out_size=5) 
T = Trainer(D) 
T.setup(model,learning_rate=1e-2, weight_decay=1e-3, max_epochs=100, batch_size=700)
T.train_model(model)
plot_train([T])
test_accuracy([D], [model])

#%%
# TODO: To look at some simple transforms, set transforms=True
# TODO: Look at longer windows of data, maybe width=64

