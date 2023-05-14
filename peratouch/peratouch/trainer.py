# Script to store all of the main functions for cleaning and loading data
import numpy as np
import torch
from torch.utils.data import DataLoader
from peratouch.data import Data
from peratouch.results import Results
# Plotting imports
from peratouch.plot import plot_trainer
import time

class Trainer:

    def __init__(self, Data : Data):
        """Links traininer to a given dataset"""
        self.Data = Data
    
    def setup(self, model, batch_size, learning_rate=1e-2, weight_decay=1e-3, max_epochs=20, verbose=True):
        
        """Setup of hyperparameters used during training."""

        self.max_epochs = max_epochs
        # Build data loader to seperate data into batches
        # n_batches = 10    # Number of minibatches to train
        # batch_size = int(len(self.Data.trainset)/n_batches)
        # print("Batch size:", batch_size)
        self.train_loader = DataLoader(self.Data.trainset, batch_size=batch_size, shuffle=True)
        # Use same criterion for all models
        self.criterion = torch.nn.CrossEntropyLoss()
        #Choose the Adam optimiser
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize lists to track progress of training 
        self.losses = []        # Track loss function
        self.accuracies = []    # Track train, validation and test accuracies
        self.epochs = []        # To track progress over epochs
        self.times = []

        self.val_loss_min = np.inf   # Used to store minimul val loss during training
        self.verbose = verbose
        self.model_name = model.__class__.__name__

    def train_model(self, model):
        """
        Loop over epochs and batches to train the model.
        Uses the hyperparameters defined in setup().
        Stores model performance at the end of each epoch. 
        Sets model to lowest validation loss achieved.
        """
        print("\n Start of training model:\n")

        model = model.to(self.Data.device)
        model.apply(weight_init)    # Fixed initialization for reproducibiity

        for epoch in range(1, self.max_epochs+1):  # Loop over the dataset multiple times

            for (sig, y) in self.train_loader:   # sig and y are batches 
                model.train() # Explicitly set to model to training 

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(sig)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            self.evaluate_model(epoch, model)

        print("\nTraining Complete!")
        print(f"Loading best weights for lowest validation loss={self.val_loss_min:.3f} ...")
        model.load_state_dict(torch.load('./state_dict.pt'))
        print(f"\nAverage running time per epoch: {np.mean(self.times):.2f} seconds")
        print(f"Total running time: {np.sum(self.times):.2f} seconds")

        # Transform into arrays
        self.losses = np.array(self.losses)
        self.accuracies = np.array(self.accuracies)

    def evaluate_model(self, epoch, model):
        """
        Calculates training and validation accuracy and losss. 
        Stores this information for the epoch given. 
        """

        # NOTE: The command below is having some really strange behaviour
        # If uncommented, it causes big deteoration in performance, and very rocky learning 
        # I believe this is a bug from pytorch, so while this is not fixed, DO NOT UNCOMMENT
        # model.eval()   # Disables some layers such as drop-out and batchnorm

        R = Results(self.Data, model)
        val_loss = R.loss_val(self.criterion)
        tr_loss = R.loss_tr(self.criterion)
        tr_acc = R.acc_tr() 
        val_acc = R.acc_val() 

        # At the end of each epoch, evaluate model 
        self.losses.append([tr_loss, val_loss])
        self.accuracies.append([tr_acc, val_acc])
        self.epochs.append(epoch)

        if (self.verbose) & (((epoch-1)%np.ceil(self.max_epochs/10))==0):       # Print 10 lines during training
            print(
                f"End of epoch {epoch}: " \
                f"loss_tr={tr_loss:5.3f}, " \
                f"loss_val={val_loss:5.3f}, " \
                f"train={tr_acc*100:4.1f}%, " \
                f"val={val_acc*100:4.1f}%"
                )

        # Store state of minimum validation loss
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), './state_dict.pt')
            self.val_loss_min = val_loss   # Update

        # Store running time of each epoch
        if epoch==1:                # Start recording at first epoch
            self.t = time.time() 
        else:
            self.times.append(time.time() - self.t)
            self.t = time.time()

    def plot_train(self, save_path=None):
        """ Plot accuracies and losses during training of the model """
        plot_trainer(self.epochs, self.losses, self.accuracies, self.model_name, save_path)


def weight_init(m):
    """
    Method to insure that weights of each layer are initialized always to 
    the same values for reproducibiity
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.manual_seed(180200742)
        torch.nn.init.kaiming_normal_(m.weight)     # Read somewhere that Kaiming initialization is advisable
        torch.nn.init.zeros_(m.bias)


