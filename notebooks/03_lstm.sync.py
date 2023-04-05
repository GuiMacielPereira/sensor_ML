#%%
# A notebook for simple lstm exploration
# Case of single lstm cell
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import lstm
from peratouch.config import datapath_five_users

input_size = 4 
D = Data(datapath_five_users, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=False)
D.tensors_to_device()
D.print_shapes()
model = lstm(input_size=input_size, hidden_size=16, out_size=5, global_pool=True) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# Look at 3 triggers
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import lstm
from peratouch.config import datapath_five_users

D = Data(datapath_five_users, triggers=True, releases=False)
D.split()
D.normalize()
D.resample_random_combinations(aug_factor=2)
D.tensors_to_device()
D.print_shapes()
#%%
model = lstm(input_size=32, hidden_size=5, out_size=5) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=5000, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# --------- Cells with some scribles ------------

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
import numpy as np
x = np.arange(50).reshape((5, 1, 10))
print(x)
res = []
input_size = 5
for i in range(x.shape[-1] - input_size + 1):
    res.append(x[:, :, i:i+input_size])
x = np.concatenate(res, axis=1)
print("Reshaped:\n", x)
print(x.shape)

