
#%%
# A notebook for simple lstm exploration
import torch.nn as nn
from torch.autograd import Variable 
import torch
from core_functions import SensorSignals
dataPath = "./second_collection_triggs_rels_32.npz"
S = SensorSignals(dataPath)  
S.split_data()
S.norm_X()
S.setup_tensors()
S.print_shapes()

#%%
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        
        self.input_size = 1 
        self.hidden_dim = 100
        self.out_size = 5
        self.num_layers = 1

        # shape of input ()
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.out_size)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((x.shape[0], x.shape[2], 1))

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
 

models = [lstm()]
S.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=256, max_epochs=10)

#%%
S.plot_train()
S.bestModelAcc()


