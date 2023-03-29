
#%%
# A notebook for simple lstm exploration
# Case of single lstm cell
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import lstm

dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=8)
D.tensors_to_device()
D.print_shapes()
model = lstm(input_size=8, hidden_size=8, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# Look at CNN-LSTM 
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import cnn_lstm

dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=8)
D.tensors_to_device()
D.print_shapes()

model = cnn_lstm(input_size=8, hidden_conv=8, hidden_lstm=8, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200)
T.train_model(model)
plot_train([T])
test_accuracy([D], [model])

#%%
# Look at longer intervals 
# Prepare data for lstm
import numpy as np
filename = "second_collection"
data = np.load((filename+".npz"))

# TODO: Split into train, val, test and do a sliding window on train as data augmentation

#%%
# Look at 3 triggers
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import lstm
dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.resample_random_combinations()
D.tensors_to_device()
D.print_shapes()
#%%
model = lstm(input_size=32, hidden_size=5, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# Look at simpler cnn_lstm 
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy
from networks import cnn_lstm_simpler
dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
model = cnn_lstm_simpler(n_ch=2, hidden_lstm=8, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# This function can be used to look at a sliding window 
# Not better performing than just using a split (i.e. no stride==input_size)
def change_input(x, I, S):
    """I is window size, S is stride"""

    # Make a mask used to transform inputs 
    bs = x.shape[0]
    W = x.shape[-1] 
    L = int((W - I) / S + 1)

    mask = np.full((L, W), False)
    mask[0, :I] = True 
    for i in range(1, L):
        mask[i] = np.roll(mask[i-1], shift=S)

    result = np.zeros((bs, L, I))
    x = x * np.ones((1, L, 1))  # multiply by ones to extend shape
    for i in range(bs):
        result[i] = x[i][mask].reshape((L, I))
    return result


#%%
# Exploring some reshaping

import torch 
x = torch.arange(30).reshape((5, 3, 2))
print(x)
x = x.transpose(2, 1)
print("Reshaped:\n", x)
