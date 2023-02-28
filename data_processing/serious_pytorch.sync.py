# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

# %%
# Tried a few things:
# More convolution layers did not increase accuracy
# BatchNorm helps the training initialy

# %%
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense

# %%
dataPath = "./second_collection_triggs_rels_32.npz"
S = SensorSignals(dataPath) 
S.split_data()
S.norm_X()
S.setup_tensors()
S.print_shapes()
S.plot_data()

#%%
# for CNN_STANDARD
# lr=5e-3, wd=1e-4
# with BatchNorm1d
# lr=1e-2, wd=1e-3

models = [CNN_Simple(input_ch=1, n_filters=8)]
S.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=256, max_epochs=50)

#%%
S.plot_train()
S.bestModelAcc()

#%%
D = SensorSignals("./second_collection_triggs_rels_32.npz") 
D.split_data()
D.norm_X()
D.resample_channels()
D.setup_tensors()
D.print_shapes()
D.plot_data()

#%%
models = [CNN_Simple(input_ch=3, n_filters=16)]
D.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=128, max_epochs=5)

#%%
D.plot_train()
D.bestModelAcc()


#%%
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense
# Look into using triggers and releases in two separate channels
E = SensorSignals("./second_collection_triggs_rels_32.npz", triggers=True, releases=False, transforms=True) 
E.split_data()
E.norm_X()
E.setup_tensors()
E.print_shapes()
E.plot_data()

#%%
models = [CNN_Simple(input_ch=2, n_filters=16)]
E.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=100)

#%%
E.plot_train()
E.bestModelAcc()

#%%
# Longer intervals of time
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense

F = SensorSignals("./second_collection_zeros_out_long_data_1024.npz")
F.split_data()
F.norm_X()
F.setup_tensors()
F.print_shapes()

#%%
models = [CNN_Simple(input_ch=1, n_filters=16)]
F.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=6*256, max_epochs=10)

#%%
F.plot_train()
F.bestModelAcc()
