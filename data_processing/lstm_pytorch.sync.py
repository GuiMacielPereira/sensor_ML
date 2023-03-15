
#%%
# A notebook for simple lstm exploration
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
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        
        torch.manual_seed(180200742)    # Set seed for same initialization of weigths each time

        self.input_size = 4 
        self.hidden_dim = 64 
        self.out_size = 5
        self.num_layers = 1

        # shape of input ()
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.out_size)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((x.shape[0], int(x.shape[2]/self.input_size), self.input_size))

        # # Initialize hidden state
        # h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)) #hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)) #internal state

        # Propagate input through LSTM
        # print("input: ", x.shape)
        lstm_out, _ = self.lstm(x) #lstm with input, hidden, and internal state
        # print("out lstm: ", lstm_out.shape)

        lstm_out = lstm_out[:, -1, :]    # Choose final output of lstm for classification

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        out = self.relu(out)
        out = out.view(batch_size, -1)

        # print("output shape: ", out.shape)
        return out 
 

model = lstm() 
T = Trainer(D)
T.setup(model, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=200)
T.train_model(model)
plot_train([T])
test_accuracy(D, [model])

#%%
import numpy as np

A = np.arange(32)
idxs = np.arange(len(A), step=2)
print(A)
print(idxs)

L = 15
M = np.zeros((L, 4)) 
for i, idx in enumerate(idxs[:-1]):
    M[i] = A[idx:idx+4]
print(M)

# Make a mask to select positions
mask = np.full((L, 32), False)
mask[0, :4] = True
for i in range(1, len(mask)):
    mask[i] = np.roll(mask[i-1], shift=2)
print(mask)

A = A[np.newaxis, :] * np.ones((15, 1))
print(A[mask].reshape((15, 4)))

