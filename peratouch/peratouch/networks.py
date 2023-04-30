
import torch.nn as nn

# ------- CNN ----------

# Standard model with usual halving of image size and doubling the depth at each conv layer
class CNN(nn.Module):    

    def __init__(self, n_ch, n_filters=8, im_size=32, out_size=5):
        """
        CNN Arquitecture: 3 convolutional layers and 2 linear onesself.
        Inputs:
            n_ch: number of input channels (number of triggers)
            n_filters: number of output channels of first conv layer
            im_size: size of trigger, 32 by default
            out_size: number of classes
        """
        super(CNN, self).__init__()

        k = n_filters   # Rename for convinience
        flat_size = int(4*k*im_size/8)   # Size after conv layers

        self.cnn = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(n_ch, k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Conv1d(k, 2*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(2*k),
            nn.ReLU(),
            nn.Conv1d(2*k, 4*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4*k),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, flat_size),    # Size of image 32 is 4
            nn.ReLU(),
            nn.Linear(flat_size, out_size)
        )

    def forward(self, x):
        return self.cnn(x)
    
# ------ LSTM --------

# Define standard lstm model
from torch import manual_seed
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size=5, global_pool=True):
        super(LSTM, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time
        self.global_pool = global_pool

        # Input (batch_size, n_sequence, input_size)
        self.lstm = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden_size, 
                batch_first=True,
                num_layers=1, 
                dropout=0, 
                bidirectional=False
                )
        self.fc = nn.Linear(hidden_size, out_size)

        # self.relu = nn.ReLU()    # Sigmoid did not work well
        
    def forward(self, x):      
        x, _ = self.lstm(x) 
        # x = self.relu(x)      # Seems to slow down training

        if self.global_pool:
            x = x.max(dim=1)[0]
        else:
            x = x[:, -1, :]
        
        return self.fc(x)         


# --------- CNN-LSTM -------------

# Standard CNN + LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, n_ch, n_filters=8, hidden_lstm=8, out_size=5, global_pool=True):
        super(CNN_LSTM, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time

        # Same CNN as before
        k = n_filters
        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(n_ch, k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Conv1d(k, 2*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(2*k),
            nn.ReLU(),
            nn.Conv1d(2*k, 4*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4*k),
            nn.ReLU(),
        )
        # Extra LSTM layer
        self.global_pool = global_pool
        self.lstm = nn.LSTM(input_size=4*k, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_lstm, out_size)
        self.relu = nn.ReLU()    # Interestingly, using Sigmoid prevents learning 
        
    def forward(self, x):
        x = self.conv(x)
        # Input size of lstm is the same as number of channels
        x = x.transpose(2, 1)    # Transpose between second and third axes
        x, _ = self.lstm(x) 

        if self.global_pool:
            return self.fc(x.max(dim=1)[0])     # Pool over all cells
        else:
            return self.fc(x[:, -1, :])         # Select only result from last cell


# NOTE: Model below works as well but abandoned it due to worse performance
# Also it operates on a very different way than conceptually stacking CNN + LSTM
# Did not find any analog in literature
# Try out a time-distributed CNN-LSTM model
class cnn_lstm_time_distributed(nn.Module):
    def __init__(self, input_size, out_size=5, global_pool=False):
        super(cnn_lstm_time_distributed, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time
        # shape of input (batch_size, n_sequence, input_size)

        self.global_pool = global_pool
        ks = 3   # kernel_size
        s = 2    # stride
        n_features = int(input_size/2)   # Number of features out of conv, input size of LSTM 
        hidden_ch = int(input_size/2)    # Number of channels out of conv 
        hidden_lstm = int(input_size/2)  # Size of output of lstm 

        self.conv = nn.Sequential(
                nn.Conv1d(1, out_channels=hidden_ch, kernel_size=ks, stride=s),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_ch*int((input_size-ks)/s + 1), n_features)
                )
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_lstm, out_size)
        self.relu = nn.ReLU()    # Interestingly, using Sigmoid prevents learning 
        
    def forward(self, x):
        bs, n_seq, input_size = x.shape
        # Prepare to run cnn along all time segments simultaneously
        x = x.view(bs*n_seq, 1, input_size)
        x = self.conv(x)
        # Bring shape back to original
        x = x.view(bs, n_seq, -1)

        x, _ = self.lstm(x) 

        if self.global_pool:
            return self.fc(x.max(dim=1)[0])     # Pool over all cells
        else:
            return self.fc(x[:, -1, :])         # Select only result from last cell

