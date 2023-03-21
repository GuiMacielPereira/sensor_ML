
#%%
# Define simplest lstm model
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(lstm, self).__init__()
        torch.manual_seed(180200742)    # Set seed for same initialization of weigths each time
        # shape of input (batch_size, n_sequence, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()    # Interestingly, using Sigmoid prevents learning 
        
    def forward(self, x):
        x, _ = self.lstm(x) 
        x = x[:, -1, :]    # Choose only output of last lstm cell for classification
        x = self.fc(x)
        x = self.relu(x)  # x.shape = (batch_size, n_classes)
        return x 

#%%
# A notebook for simple lstm exploration
# Case of single lstm cell
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy

dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()
D.tensors_to_device()
D.print_shapes()
#%%
model = lstm(input_size=32, hidden_size=8, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])


#%%
# Split input into several windows
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import Data, Trainer, plot_train, test_accuracy
import numpy as np

dataPath = "./second_collection_triggs_rels_32.npz"
D = Data(dataPath, triggers=True, releases=False)
D.split()
D.normalize()

def reshape_seq(x, input_size):
    if x.shape[-1] % input_size: raise ValueError("Splitting size not matching!")
    x = x.reshape((x.shape[0], int(x.shape[-1]/input_size), input_size))
    return x

input_size = 8 

D.Xtrain = reshape_seq(D.Xtrain, input_size) 
D.Xtest = reshape_seq(D.Xtest, input_size) 
D.Xval = reshape_seq(D.Xval, input_size) 

D.tensors_to_device()
D.print_shapes()

#%%
# Try out a simple CNN + LSTM model
class cnn_lstm(nn.Module):
    def __init__(self, input_size, hidden_conv, hidden_lstm, out_size):
        super(cnn_lstm, self).__init__()
        torch.manual_seed(180200742)    # Set seed for same initialization of weigths each time
        # shape of input (batch_size, n_sequence, input_size)
        ks = 3   # kernel_size
        s = 2    # stride
        n_features = 8 

        self.conv = nn.Sequential(
                nn.Conv1d(1, out_channels=hidden_conv, kernel_size=ks, stride=s),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_conv*int((input_size-ks)/2 + 1), n_features)
                )

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_lstm, out_size)
        self.relu = nn.ReLU()    # Interestingly, using Sigmoid prevents learning 
        
    def forward(self, x):
        bs, n_seq, input_size = x.shape
        x = x.view(bs*n_seq, 1, input_size)
        x = self.conv(x)
        x = x.view(bs, n_seq, -1)

        x, _ = self.lstm(x) 
        x = x[:, -1, :]    # Choose only output of last lstm cell for classification
        x = self.fc(x)
        x = self.relu(x)  # x.shape = (batch_size, n_classes)
        return x 


#%%
model = cnn_lstm(input_size, hidden_conv=8, hidden_lstm=8, out_size=3) 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])


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
