# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

# %%
# Tried a few things:
# More convolution layers did not increase accuracy
# BatchNorm helps the training initialy

#%%
from core_functions import Data, Trainer, plot_train
from networks import CNN_Simple, CNN_Dense
dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()

#%%
model = CNN_Simple(input_ch=1, n_filters=8)
T = Trainer(D)
T.setup(model)
T.train_model(model)
plot_train([T])
print(f"Test Accuracy of best state model: {D.acc_te(model)*100:.1f}")

# %%
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense
dataPath = "./second_collection_triggs_rels_32.npz"
S = SensorSignals(dataPath, triggers=True, releases=True) 
S.split_data()
S.norm_X()
S.setup_tensors()
S.print_shapes()
S.plot_data()
#%%
models = [CNN_Simple(input_ch=2, n_filters=8)]
S.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=50)
#%%
S.plot_train()
S.bestModelAcc()

#%%
# Look at using several signals, one per channel
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense
D = SensorSignals("./second_collection_triggs_rels_32.npz", triggers=True, releases=False) 
D.split_data()
D.norm_X()
D.resample_random_combinations()
D.setup_tensors()
D.print_shapes()
# D.plot_data()
#%%
models = [CNN_Simple(input_ch=3, n_filters=16)]
D.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=10*256, max_epochs=30)
#%%
D.plot_train()
D.bestModelAcc()


#%%
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense
# Look into transformations 
E = SensorSignals("./second_collection_triggs_rels_32.npz", triggers=True, releases=False, transforms=True) 
E.split_data()
E.norm_X()
E.setup_tensors()
E.print_shapes()
E.plot_data()
#%%
models = [CNN_Simple(input_ch=2, n_filters=16)]
E.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=50)
#%%
E.plot_train()
E.bestModelAcc()

#%%
# Longer intervals of time
from core_functions import SensorSignals
from networks import CNN_Simple, CNN_Dense, CNN_64
F = SensorSignals("./second_collection_triggs_rels_64.npz")
F.split_data()
F.norm_X()
F.setup_tensors()
F.print_shapes()
#%%
models = [CNN_64(input_ch=1, n_filters=8)]
F.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=256, max_epochs=100)
#%%
F.plot_train()
F.bestModelAcc()
