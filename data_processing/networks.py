
import torch.nn as nn

# Model with usual halving of image size and doubling the depth
class CNN_Simple(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=32, out_size=3):
        """input_ch is number of channels in initial image, n_filters is first number of filters."""
        super(CNN_Simple, self).__init__()

        k = n_filters

        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(input_ch, k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Conv1d(k, 2*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(2*k),
            nn.ReLU(),
            nn.Conv1d(2*k, 4*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4*k),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(        # Fully connected part, 3 layers
            nn.Linear(4*k * int(im_size/8), 256),    # Size of image 32 is 4
            nn.ReLU(),
            nn.Linear(256, out_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# Same network as above, but with extra convolutional layers
class CNN_Dense(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=32, out_size=3):
        """input_ch is number of channels in initial image, n_filters is first number of filters."""
        super(CNN_Dense, self).__init__()

        k = n_filters

        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(input_ch, k, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(k, k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Conv1d(k, 2*k, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(2*k, 2*k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2*k),
            nn.ReLU(),
            nn.Conv1d(2*k, 4*k, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(4*k, 4*k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4*k),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(        # Fully connected part, 3 layers
            nn.Linear(4*k * int(im_size/8), 256),    # Size of image 32 is 4
            nn.ReLU(),
            nn.Linear(256, out_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# Define simplest lstm model
from torch import manual_seed
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(lstm, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time
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

# Try out a simple CNN + LSTM model
class cnn_lstm(nn.Module):
    def __init__(self, input_size, hidden_conv, hidden_lstm, out_size):
        super(cnn_lstm, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time
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

# Try out a simple CNN + LSTM model
class cnn_lstm_simpler(nn.Module):
    def __init__(self, n_ch, hidden_lstm, out_size):
        super(cnn_lstm_simpler, self).__init__()
        manual_seed(180200742)    # Set seed for same initialization of weigths each time

        # Same CNN as before
        k = n_ch
        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(1, k, kernel_size=3, stride=2, padding=1),
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
        self.lstm = nn.LSTM(input_size=4*k, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_lstm, out_size)
        self.relu = nn.ReLU()    # Interestingly, using Sigmoid prevents learning 
        
    def forward(self, x):
        x = self.conv(x)
        # Input size of lstm is the same as number of channels
        x = x.transpose(2, 1)    # Transpose between second and third axes
        x, _ = self.lstm(x) 
        x = x[:, -1, :]    # Choose only output of last lstm cell for classification
        x = self.fc(x)
        x = self.relu(x)  # x.shape = (batch_size, n_classes)
        return x 

# Not used in a long time, only for triggers with size 64
class CNN_64(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=64, out_size=3):
        """input_ch is number of channels in initial image, n_filters is first number of filters."""
        super(CNN_64, self).__init__()

        k = n_filters

        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(input_ch, k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Conv1d(k, 2*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(2*k),
            nn.ReLU(),
            nn.Conv1d(2*k, 4*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(4*k),
            nn.ReLU(),
            nn.Conv1d(4*k, 8*k, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(8*k),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(        # Fully connected part, 3 layers
            nn.Linear(8*k * int(im_size/16), 256),    # Size of image 32 is 4
            nn.ReLU(),
            nn.Linear(256, out_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
