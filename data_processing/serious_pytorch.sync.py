# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

#%%
# Example using several models 
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import CNN_Simple, CNN_Dense
# dataPath = "./second_collection_triggs_rels_32.npz"
dataPath = "./five_people_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
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
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import CNN_Simple, CNN_Dense
# dataPath = "./second_collection_triggs_rels_32.npz"
dataPath = "./five_people_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=True)
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
# Look at using several signals, one per channel
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import CNN_Simple, CNN_Dense
# dataPath = "./second_collection_triggs_rels_32.npz"
dataPath = "./five_people_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
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

# model = CNN_Simple(input_ch=3, n_filters=16)
# T = Trainer(D)
# T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*5*256, max_epochs=30)
# T.train_model(model)
plot_train(trainers)
test_accuracy(D, models)


#%%
# Look at some simple transforms
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import CNN_Simple, CNN_Dense
dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False, transforms=True)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
D.plot_data()
#%%
model = CNN_Simple(input_ch=2, n_filters=16)
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=50)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])


#%%
# # Longer intervals of time
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import CNN_Simple, CNN_64 
dataPath = "./second_collection_triggs_rels_64.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
model = CNN_64(input_ch=1, n_filters=16)
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=50)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])

