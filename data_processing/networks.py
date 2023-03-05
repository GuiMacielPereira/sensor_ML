
import torch.nn as nn

# Model with usual halving of image size and doubling the depth
class CNN_Simple(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=32):
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
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# Same network as above, but with extra convolutional layers
class CNN_Dense(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=32):
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
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class CNN_64(nn.Module):    

    def __init__(self, input_ch, n_filters, im_size=64):
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
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
