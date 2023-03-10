# Script to store all of the main functions for cleaning and loading data
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Data:
    def __init__(self, dataPath, triggers=True, releases=False, transforms=False):

        length = int(dataPath.split("_")[-1].split(".npz")[0])
        if length >= 1024:
            self.Xraw, self.yraw = load_long_data(dataPath)
        else:
            self.Xraw, self.yraw = load_short_data(dataPath, triggers, releases, transforms)


    def split(self):
        Xtrain, self.Xtest, ytrain, self.ytest = train_test_split(self.Xraw, self.yraw, test_size=0.15, random_state=42)
        self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(Xtrain, ytrain, test_size=0.15, random_state=42)

    def normalize(self):
        """Normalise datasets according to fixed value from train set"""
        # Fix normalisation value
        xmax = np.mean(np.max(self.Xtrain, axis=-1, keepdims=True), axis=0, keepdims=True)     # Hard coding the normalization severely affects validation accuracy

        def norm(x):
            x /= xmax
            print(f"{np.mean(np.max(x, axis=-1), axis=0)}")

        print("Train, test and validation data normalized to:")
        norm(self.Xtrain)
        norm(self.Xtest)
        norm(self.Xval)

    def resample_random_combinations(self):
        """Makes random combinations of a single channel"""

        np.random.seed(0)
        def make_combinations(X):
            return resample_with_replacement(X, n_channels=3, no_combinations=5*len(X))

        self.Xtrain, self.ytrain = resample_by_user(make_combinations, self.Xtrain, self.ytrain)
        # Data augmentation should not be performed on test and validation set
        # Should be resampled_without_replacement
        self.Xtest, self.ytest = resample_by_user(make_combinations, self.Xtest, self.ytest)
        self.Xval, self.yval = resample_by_user(make_combinations, self.Xval, self.yval)

    def resample_trigs_rels(self):
        """Assigns triggers to releases randomly."""
        # Generally not used, no advantage observed from this type of data augmentation

        np.random.seed(0)
        def make_combinations(X):
            return resample_trigs_rels(X, no_combinations=5*len(X))
        self.Xtrain, self.ytrain = resample_by_user(make_combinations, self.Xtrain, self.ytrain)

    def plot_data(self):
        n_users = len(np.unique(self.ytrain))
        n_ch = self.Xtrain.shape[1]
        plt.figure(figsize=(n_users*5, n_ch*5))
        plt.suptitle("Mean and std of signals for users and channels")
        plt.tight_layout()
        for i, u in enumerate(np.unique(self.ytrain)):
            X = self.Xtrain[self.ytrain==u]
            Xmean = np.mean(X, axis=0, keepdims=True)
            Xstd = np.std(X, axis=0, keepdims=True)
            for j, (mean, std) in enumerate(zip(Xmean[0], Xstd[0])):
                plt.subplot(n_ch, n_users, n_users*j + i+1)
                plt.title(f"user={i} ch={j}")
                plt.errorbar(np.arange(len(mean)), mean, std, fmt="b.")
                plt.xticks([])

    def tensors_to_device(self):
        # Use GPU if available 
        self.device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
        self.dtype = torch.float32
        print("Using Device: ", self.device, ", dtype: ", self.dtype)

        def to_tensor(X, y):
            xt = torch.tensor(X, dtype=self.dtype).to(self.device)
            yt = torch.tensor(y, dtype=torch.long).to(self.device)
            return xt, yt 
        
        self.xtr, self.ytr = to_tensor(self.Xtrain, self.ytrain)
        self.xv, self.yv = to_tensor(self.Xval, self.yval)
        self.xte, self.yte = to_tensor(self.Xtest, self.ytest)

        # Create trainset in the correct format for dataloader
        self.trainset = [[x, y] for (x, y) in zip(self.xtr, self.ytr)]

    def print_shapes(self):
        print("Raw data shape: ", self.Xraw.shape)
        print("Labels shape: ", self.yraw.shape)
        print("Unique labels: ", np.unique(self.yraw))
        print("Shape of test set:", self.Xtest.shape)
        print("Shape of train set:", self.Xtrain.shape)
        print("Shape of validation set:", self.Xval.shape)
        print("Fraction of single class in test set: ", np.mean(self.ytest==0))
        print("dtype of inputs: ", self.xtr.dtype)

    def acc_tr(self, model):
        return acc(model, self.xtr, self.ytr)

    def acc_val(self, model):
        return acc(model, self.xv, self.yv)

    def acc_te(self, model):
        return acc(model, self.xte, self.yte)

    def loss_val(self, model, criterion):
        with torch.no_grad():    # Each time model is called, need to avoid updating the weights
            return criterion(model(self.xv), self.yv).item()

def acc(model, x, y):
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out.data, 1)
        return (pred==y).detach().cpu().numpy().mean()


class Trainer:

    def __init__(self, D:Data):
        self.Data = D

    
    def setup(self, model, batch_size=256, learning_rate=1e-2, weight_decay=1e-3, max_epochs=20):
        self.max_epochs = max_epochs

        # Build data loader to seperate data into batches
        self.train_loader = DataLoader(self.Data.trainset, batch_size=batch_size, shuffle=True)
        # Use same criterion for all models, cross entropy is good for classification problems
        self.criterion = torch.nn.CrossEntropyLoss()
        #Choose the Adam optimiser
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize lists to track progress of training 
        self.losses = []        # Track loss function
        self.accuracies = []    # Track train, validation and test accuracies
        self.epochs = []        # To track progress over epochs

    def train_model(self, model):

        model = model.to(self.Data.device)
        model.apply(weight_init)    # Fixed initialization for reproducibiity
        val_loss_min = np.inf

        for epoch in range(self.max_epochs):  # Loop over the dataset multiple times

            for (sig, y) in self.train_loader:   # sig and y are batches 
                model.train() # Explicitly set to model to training 

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(sig)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                # Print and store results
                # if i % 100 == 0:
            # At the end of each epoch, evaluate model 
            model.eval()   # Disables some layers such as drop-out and batchnorm
            val_loss = self.Data.loss_val(model, self.criterion)
            losses = [loss.item(), val_loss]
            acc = [self.Data.acc_tr(model), self.Data.acc_val(model)] 
            print(f"End of epoch {epoch+1}: loss_tr={losses[0]:5.3f}, loss_val={losses[1]:5.3f}, train={acc[0]*100:4.1f}%, val={acc[1]*100:4.1f}%")
            self.losses.append(losses)
            self.accuracies.append(acc)
            self.epochs.append(epoch+1)

            if val_loss < val_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                val_loss_min = val_loss   # Update

        print("Training Complete!")
        print(f"Loading best weights for lowest validation loss={val_loss_min:.3f} ...")
        model.load_state_dict(torch.load('./state_dict.pt'))

        self.losses = np.array(self.losses)
        self.accuracies = np.array(self.accuracies)
        return  self.losses, self.accuracies

def weight_init(m):
    """
    Method to insure that weights of each layer are initialized always to 
    the same values for reproducibiity
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.manual_seed(180200742)
        torch.nn.init.kaiming_normal_(m.weight)     # Read somewhere that Kaiming initialization is advisable
        torch.nn.init.zeros_(m.bias)


    # def train_multiple_models(self, models, learning_rate, weight_decay, batch_size, max_epochs):
    #
    #     # Transform single values into arrays
    #     def toArray(arg):
    #         if ~isinstance(arg, list):
    #             return np.full(len(models), arg)
    #         return arg
    #     
    #     learning_rate = toArray(learning_rate)
    #     weight_decay = toArray(weight_decay)
    #
    #
    #     self.models_label = [f"model {i}" for i in range(len(models))]
    #     self.models = models
    #     self.models_loss = [] 
    #     self.models_acc = [] 
    #     self.models_best_states = []
    #
    #     for model, lr, wd  in zip(self.models, learning_rate, weight_decay):
    #         
    #         self.train_model(model, learning_rate=lr, weight_decay=wd, batch_size=batch_size, max_epochs=max_epochs )
    #
    #         self.models_loss.append(self.losses)
    #         self.models_acc.append(self.accuracies)
    #         self.models_best_states.append(self.best_state)

def plot_train(trainers):
    """ Plot accuracies and losses durin hte training of the model """
    plt.figure(figsize=(8, 5))
    for i, T in enumerate(trainers):
        for data, lab in zip([T.accuracies, T.losses], ["Accuracy", "Loss"]):
            plt.plot(T.epochs, data, label=[f"model {i}, Train "+lab, f"model {i}, Val "+lab])
            
    # for models_metric, ylabel in zip([self.models_acc, self.models_loss], ["Accuracy", "Loss"]):
    #     for lab, accs in zip(self.models_label, models_metric):
    #         plt.plot(self.epochs, accs, label=[lab+", Train "+ylabel, lab+", Val "+ylabel])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylim(0, 1)
    
# def bestModelAcc(self):
#     """
#     Prints test accuracy of best model
#     Chooses model that yields the best validation accuracy
#     S is object containing the data used during training 
#     """
#     for m, state in zip(self.models, self.models_best_states):
#         m.load_state_dict(state)
#
#     best_accs = [self.acc_te(model) for model in self.models]
#     # best_loss = [self.loss_val(model) for model in self.models]
#     for model, acc in zip(self.models, best_accs):
#         print(f"Test accuracy of lowest val acc={self.acc_val(model)*100:.1f}: {acc*100:.1f}%")
#
    # best_acc_idx = np.argmax([acc[-1, -1] for acc in self.models_acc])
    # best_model = self.models[best_acc_idx]
    # best_acc = self.acc_te(best_model)
    # print(f"Accuracy of test set of best model (idx={best_acc_idx}): {best_acc*100:.1f}%")


# Functions used in the class above 
def resample_by_user(make_combinations, X, y):
    """
    Takes in a given procedure for making combinations and 
    applies it to each individual user
    """

    newX = []
    newy = []
    for u in np.unique(y):    # Loop over all users 

        Xuser = X[y==u]    
        Xcomb = make_combinations(Xuser)

        newX.append(Xcomb)
        newy.append(np.full(len(Xcomb), u))

    return np.concatenate(newX), np.concatenate(newy)


def resample_with_replacement(X, n_channels, no_combinations):
    """From X.shape[0] choose n_channels, repeated no_combinations times."""

    result = np.zeros((no_combinations, n_channels, *X.shape[1:]))
    for i in range(no_combinations):
        result[i] = X[np.random.randint(0, X.shape[0], size=n_channels)]   # index 0 to match shape

    # Reshape into correct number of channels.
    # Accounts for case where both triggers and releases are considered.
    return result.reshape((no_combinations, n_channels*X.shape[1], *X.shape[2:]))


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
    

def load_short_data(dataPath, triggers=True, releases=False, transforms=False):

    assert (triggers or releases), "At least one of triggers or releases need to be set to True!"
    data = np.load(dataPath)

    # Check different users
    users = np.unique(np.array([key.split("_")[0] for key in data], dtype=str))

    # Build X data and corresponding labels
    Xraw = []
    yraw = []

    for u in users:

        userX = []

        if triggers:
            userX.append(data[u+"_triggers"])
            if transforms:
                Xt = np.diff(data[u+"_triggers"], axis=-1)
                userX.append(np.pad(Xt, pad_width=((0, 0), (0, 1))))   # Add zeros to end of each signal to match shape
                # Xt = np.diff(data[u+"_triggers"], n=2, axis=-1)
                # userX.append(np.pad(Xt, pad_width=((0, 0), (1, 1))))
        if releases:
            userX.append(data[u+"_releases"])
            if transforms:
                Xt = np.diff(data[u+"_releases"], axis=-1)
                userX.append(np.pad(Xt, pad_width=((0, 0), (0, 1))))
        
        Xraw.append(np.stack(userX, axis=1))
        yraw.append(np.full(len(userX[0]), np.argwhere(users==u)[0]))


    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw


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

