
import numpy as np

# Takes in path for raw npz data. 
def prepare_lstm_long_data(datapath, stride=1024, length=1024):
    """
    Function used to create a sliding window over raw data. 
    The partition of train, test and validation is done manually without random shuffling. 
    All of the data is used, including zeros and plateaus.
    Iterates over all users and splits the signals accordingly. 
    Forms train, test and validation set with the correct labels
    """
    data = np.load(datapath)

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
    """
    Function that creates sliding window over provided signal sig. 
    Option to specify stride and length of window. 
    Option to not include zeros.
    """

    if filterZerosOut:
        sig = sig[sig>=0.01]

    idxs = np.arange(0, len(sig), step=stride)   # Discard the last index so lengths match
    X = []
    for i in idxs:
        cut = sig[i : i+length]
        if len(cut)==length:
            if np.any(cut>=2):
                X.append(cut)

    return np.vstack(X)


import matplotlib.pyplot as plt
def plot_sliding_window(dataset, n_samp=10):
    """Look at first n_samp and randomly chosen n_samp on dataset provided"""
    plt.figure()
    r_idxs = np.random.randint(0, len(dataset), size=n_samp)
    f_idxs = np.arange(0, n_samp)
    for title, idxs in zip([f"First {n_samp} collections", f"Random {n_samp} collections"], [f_idxs, r_idxs]):
        plt.figure(figsize=(8, 6))
        for i, idx in enumerate(idxs):
            s = dataset[idx][0]          # Choose data, not labels
            plt.subplot(n_samp, 1, i+1)
            plt.plot(np.arange(s.size), s, "b.")
        plt.suptitle(title)

    plt.show()

# The lstm that was used on the functions above was the following:
# This model was abandoned due to poor performance
# I also tried to remove the linear layer, but was again unable to learn.

from torch import nn, manual_seed
class lstm_many_to_many(nn.Module):
    """
    The idea behind this many-to-many architecture is that 
    the whole signal gets streteched along the lstm. 
    So each cell corresponds to a window, and the next cell to the next window, etc. 
    Each cell outputs a number of features hidden_size, which get converted to out_size 
    through a linear layer (linear weights are shared between all cells). 

    In most cases, this model failed. 
    Also I don't know if it makes sense to have a varying lstm size depending on the input. 
    For example, for training there will be thousands of cells, but if we wanted to test a single 
    window, then the lstm would only run a single cell. Doesn't seem a very good approach.
    """
    def __init__(self, input_size, hidden_size=100, out_size=3):
        super(lstm_many_to_many, self).__init__()
        manual_seed(180200742)    

        # shape of input (batch_size, n_sequence, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        x, _ = self.lstm(x.squeeze()) # out shape (n_seq, hidden_size) 
        x = self.linear(x)
        return x


# Function used for loading long windows of data
def load_long_data(dataPath):
    data = np.load(dataPath)

    Xraw = []
    yraw = []

    for i, key in enumerate(data):
        X = data[key]
        Xraw.append(X)
        yraw.append(np.full(len(X), i))

    Xraw = np.concatenate(Xraw)[:, np.newaxis, :]
    yraw = np.concatenate(yraw)
    return Xraw, yraw

# Resample triggers and releases
def resample_trigs_rels(X, no_combinations):
    """
    From X with two channels, one for trigers and another for releases,
    create random combinations between triggers and releases. 
    """

    result = np.zeros((no_combinations, 2, X.shape[-1]))
    for i in range(no_combinations):
        result[i, 0] = X[np.random.randint(0, X.shape[0]), 0] 
        result[i, 1] = X[np.random.randint(0, X.shape[0]), 1] 
    return result

# # This function was called under the Data Class:
# def resample_trigs_rels(self):
#     """Assigns triggers to releases randomly."""
#     # Generally not used, no advantage observed from this type of data augmentation
#     np.random.seed(0)
#     def make_combinations(X):
#         return resample_trigs_rels(X, no_combinations=5*len(X))
#     self.Xtrain, self.ytrain = resample_by_user(make_combinations, self.Xtrain, self.ytrain)
