# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
# Standard example 
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import CNN_Simple, CNN_Dense
from peratouch.config import datapath_five_users

D = Data(datapath_five_users, triggers=True, releases=False)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
models = [CNN_Simple(input_ch=1, n_filters=8, out_size=5), CNN_Dense(input_ch=1, n_filters=8, out_size=5)]
trainers = [Trainer(D), Trainer(D)]
for model, T in zip(models, trainers):
    T.setup(model, max_epochs=50, batch_size=2*256)
    T.train_model(model)
plot_train(trainers)
test_accuracy(D, models)

# %%
# Use both triggers and releases
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import CNN_Simple, CNN_Dense
from peratouch.config import datapath_five_users
D = Data(datapath_five_users, triggers=True, releases=True)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
D.plot_data()
#%%
model = CNN_Simple(input_ch=2, n_filters=8, out_size=5)
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=50)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])

#%%
# Look at 3 channels
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import CNN_Simple, CNN_Dense
from peratouch.config import datapath_five_users
D = Data(datapath_five_users, triggers=True, releases=False)
D.split()
D.normalize()
D.resample_random_combinations(aug_factor=2)
D.tensors_to_device()
D.print_shapes()
D.plot_data()
#%%
models = [CNN_Simple(input_ch=3, n_filters=16, out_size=5), CNN_Dense(input_ch=3, n_filters=16, out_size=5)]
trainers = [Trainer(D), Trainer(D)]
for model, T in zip(models, trainers):
    T.setup(model,learning_rate=1e-2, weight_decay=1e-3, max_epochs=20, batch_size=2*2*256)
    T.train_model(model)
plot_train(trainers)
test_accuracy(D, models)

#%%
# TODO: To look at some simple transforms, set transforms=True
# TODO: Look at longer windows of data, maybe width=64

