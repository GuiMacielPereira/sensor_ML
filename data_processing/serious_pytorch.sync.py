# %%
# Notebook to explore more serious convolutional networks 
# i.e. includes analysis of training and test accuracies

# %%
# Tried a few things:
# More convolution layers did not increase accuracy
# BatchNorm helps the training initialy

# %%
import torch.nn as nn
from core_functions import SensorSignals

# Model with usual halving of image size and doubling the depth
class CNN_Best(nn.Module):    

    def __init__(self, input_ch, n_filters):
        """input_ch is number of channels in initial image, n_filters is first number of filters."""
        super(CNN_Best, self).__init__()

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
            nn.Linear(4*k * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    
class CNN_5(nn.Module):    
    def __init__(self):
        super(CNN_5, self).__init__()

        self.conv = nn.Sequential(    # Convolutional part, 3 layers
            nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(        # Fully connected part, 3 layers
            nn.Linear(16 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# %%
dataPath = "./second_collection_triggs_rels_32.npz"
S = SensorSignals(dataPath) 
S.split_data()
S.norm_X()
S.setup_tensors()
S.print_shapes()
S.plot_data()

#%%
# for CNN_STANDARD
# lr=5e-3, wd=1e-4
# with BatchNorm1d
# lr=1e-2, wd=1e-3

models = [CNN_Best(input_ch=1, n_filters=8)]
S.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=256, max_epochs=50)

#%%
S.plot_train()
S.bestModelAcc()

#%%
D = SensorSignals("./second_collection_triggs_rels_32.npz") 
D.split_data()
D.norm_X()
D.resample_channels()
D.setup_tensors()
D.print_shapes()
D.plot_data()

#%%
models = [CNN_Best(input_ch=3, n_filters=16)]
D.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=128, max_epochs=5)

#%%
D.plot_train()
D.bestModelAcc()


#%%
# Look into using triggers and releases in two separate channels
E = SensorSignals("./second_collection_triggs_rels_32.npz", triggers=True, releases=True) 
E.split_data()
E.norm_X()
# E.resample_channels()
E.setup_tensors()
E.print_shapes()
E.plot_data()

#%%
models = [CNN_Best(input_ch=2, n_filters=16)]
E.train_multiple_models(models, learning_rate=1e-2, weight_decay=1e-3, batch_size=2*256, max_epochs=100)

#%%
E.plot_train()
E.bestModelAcc()
