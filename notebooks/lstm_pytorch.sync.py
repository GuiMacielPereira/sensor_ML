#%%
# A notebook for simple lstm exploration
# Case of single lstm cell
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import lstm, lstm_pool
from peratouch.config import datapath_three_users

input_size = 4 
D = Data(datapath_three_users, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=input_size, sliding=True)
D.tensors_to_device()
D.print_shapes()
model = lstm(input_size=input_size, hidden_size=16, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# Look at CNN-LSTM 
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import cnn_lstm 
from peratouch.config import datapath_three_users

D = Data(datapath_three_users, triggers=True, releases=False)
D.split()
D.normalize()
D.reshape_for_lstm(input_size=8, sliding=True)
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
# TODO: Clean this AWFUL MESS
import numpy as np
filename = "second_collection.npz"

# TODO: Split into train, val, test and do a sliding window on train as data augmentation
def prepare_lstm_long_data(dataPath, stride=1024, length=1024):
    data = np.load(dataPath)

    Xtr, ytr = [], []
    Xte, yte = [], []
    Xval, yval = [], []

    for i, key in enumerate(data):
        udata = data[key]
        
        uXtr = sliding_window(udata[ : int(0.7*len(udata))], stride, length)
        uXte = sliding_window(udata[int(0.7*len(udata)) : int(0.85*len(udata))], stride, length)
        uXval = sliding_window(udata[int(0.85*len(udata)) : ], stride, length)

        Xtr.append(uXtr)
        ytr.append(np.full(len(uXtr), i))
        Xte.append(uXte)
        yte.append(np.full(len(uXte), i))
        Xval.append(uXval)
        yval.append(np.full(len(uXval), i))

    Xtr = np.concatenate(Xtr)[:, np.newaxis, :]
    ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte)[:, np.newaxis, :]
    yte = np.concatenate(yte)
    Xval = np.concatenate(Xval)[:, np.newaxis, :]
    yval = np.concatenate(yval)
    return Xtr, ytr, Xte, yte, Xval, yval 

def sliding_window(sig, stride, length, filterZerosOut=True):
    if filterZerosOut:
        sig = sig[sig>=0.01]

    idxs = np.arange(0, len(sig), step=stride)   # Discard the last index so lengths match
    X = []
    for i in idxs:
        cut = sig[i : i+length]
        if len(cut)==length:
            if np.any(cut>=2):
                # if np.sum(np.diff(np.argwhere(sig<=0.05)[:, 0])>1) >= 2:
                X.append(cut)

    return np.vstack(X)

# Xtr, ytr, Xte, yte, Xval, yval = prepare_lstm_long_data(filename) 
datasets = prepare_lstm_long_data(filename)
for d in datasets:
    print(d.shape)

import matplotlib.pyplot as plt
data = datasets[0]
n_samp = 10
plt.figure()
r_idxs = np.random.randint(0, len(data), size=n_samp)
f_idxs = np.arange(0, n_samp)
for title, idxs in zip([f"First {n_samp} collections", f"Random {n_samp} collections"], [f_idxs, r_idxs]):
    plt.figure(figsize=(8, 6))
    for i, idx in enumerate(idxs):
        s = data[idx][0]
        plt.subplot(n_samp, 1, i+1)
        plt.plot(np.arange(s.size), s, "b.")
    plt.suptitle(title)
plt.show()

#%%
# TODO: Need to try an LSTM layer with a batch_size=1 and each LSTM cell is a sliding window
# Then can also introduce shitfitng of training data set as data aug.
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import lstm_many_to_many 
from peratouch.config import datapath_three_users
stride, length = 100, 100
D = Data(datapath_three_users, triggers=True, releases=False)
# Nasty stuff
D.Xtrain, D.ytrain, D.Xtest, D.ytest, D.Xval, D.yval = prepare_lstm_long_data(filename, stride, length)
D.normalize()
D.tensors_to_device()
D.print_shapes()
model = lstm_many_to_many(input_size=length) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200, verbose=True)
T.train_model(model)

plot_train([T])
test_accuracy([D], [model])

#%%
# Look at 3 triggers
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import lstm
from peratouch.config import datapath_three_users

D = Data(datapath_three_users, triggers=True, releases=False)
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
from peratouch.core_funcs import Data, Trainer, plot_train, test_accuracy
from peratouch.networks import cnn_lstm_simpler 
from peratouch.config import datapath_three_users

D = Data(datapath_three_users, triggers=True, releases=False)
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

