# Script to store all of the main functions for cleaning and loading data
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from itertools import combinations
import matplotlib.pyplot as plt

class SensorSignals:

    # TODO: Clean up these initial functions to allow Xraw to have two channels: one for triggers and another for releases
    def __init__(self, dataPath, triggers=True, releases=False):
        self.Xraw, self.yraw = load_data(dataPath, triggers, releases)

    def split_data(self):
        Xtrain, self.Xtest, ytrain, self.ytest = train_test_split(self.Xraw, self.yraw, test_size=0.15, random_state=42)
        self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(Xtrain, ytrain, test_size=0.15, random_state=42)

    def set_number_channels(self, n_channels=1):

        def change_datasets(X, y):

            if n_channels==1:
                return X[:, np.newaxis, :], y
            else:
                np.random.seed(0)
                return resample_with_replacement(X, y, no_combinations=5*len(X), n_channels=3)

        self.Xtrain, self.ytrain = change_datasets(self.Xtrain, self.ytrain)
        self.Xtest, self.ytest = change_datasets(self.Xtest, self.ytest)
        self.Xval, self.yval = change_datasets(self.Xval, self.yval)


    def norm_X(self):
        """Normalise datasets according to fixed value from train set"""
        # Fix normalisation value
        xmax = np.mean(np.max(self.Xtrain, axis=-1))     # Hard coding the normalization severely affects validation accuracy

        def norm(x):
            print(f"Before: {np.mean(np.max(x, axis=-1))}")
            print(f"Normalizing dataset by {xmax:.2f}")
            x /= xmax
            print(f"After: {np.mean(np.max(x, axis=-1))}")

        norm(self.Xtrain)
        norm(self.Xtest)
        norm(self.Xval)


    def setup_tensors(self):
        # Use GPU if available 
        self.device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
        self.dtype = torch.float32
        print("Using Device: ", self.device, ", dtype: ", self.dtype)

        def toTensor(X, y):
            xt = torch.tensor(X, dtype=self.dtype)
            yt = torch.tensor(y, dtype=torch.long)
            return xt, yt 
        
        self.xtr, self.ytr = toTensor(self.Xtrain, self.ytrain)
        self.xv, self.yv = toTensor(self.Xval, self.yval)
        self.xte, self.yte = toTensor(self.Xtest, self.ytest)

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

    def weight_init(self, m):
        """
        Method to insure that weights of each layer are initialized always to 
        the same values for reproducibiity
        """
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.manual_seed(180200742)
            torch.nn.init.kaiming_normal_(m.weight)     # Read somewhere that Kaiming initialization is advisable
            torch.nn.init.zeros_(m.bias)

    def train_model(self, model, batch_size=32, learning_rate=5e-4, max_epochs=20, weight_decay=1e-4):

        model = model.to(self.device)
        model.apply(self.weight_init)    # Fixed initialization for reproducibiity
        
        self.losses = []        # Track loss function
        self.accuracies = []    # Track train, validation and test accuracies
        self.epochs = []        # To track progress over epochs

        # Build data loader to seperate data into batches
        train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        # Use same criterion for all models, cross entropy is good for classification problems
        criterion = torch.nn.CrossEntropyLoss()       

        #Choose the Adam optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(max_epochs):  # Loop over the dataset multiple times

            for i, (sig, y) in enumerate(train_loader):   # sig and y are batches 
                model.train() # Explicitly set to model to training 

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(sig)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                # Print and store results
                if i % 100 == 0:
                    acc = [self.acc_tr(model), self.acc_val(model)] 
                    print(f"Epoch {epoch+1}, Batch {i+1}: loss={loss.item():5.3f}, train={acc[0]*100:4.1f}%, val={acc[1]*100:4.1f}%")
                    self.losses.append(loss.item())
                    self.accuracies.append(acc)
                    self.epochs.append(epoch+1)

        print("Training Complete!")
        self.losses = np.array(self.losses)
        self.accuracies = np.array(self.accuracies)
        return  np.array(self.losses), np.array(self.accuracies)

    def acc(self, model, x, y):
        with torch.no_grad():
            out = model(x)
            _, pred = torch.max(out.data, 1)
            return (pred==y).detach().numpy().mean()

    def acc_tr(self, model):
        return self.acc(model, self.xtr, self.ytr)

    def acc_val(self, model):
        return self.acc(model, self.xv, self.yv)

    def acc_te(self, model):
        return self.acc(model, self.xte, self.yte)
    

    def train_multiple_models(self, models, learning_rate, weight_decay, batch_size, max_epochs):

        # Transform single values into arrays
        def toArray(arg):
            if ~isinstance(arg, list):
                return np.full(len(models), arg)
            return arg
        
        learning_rate = toArray(learning_rate)
        weight_decay = toArray(weight_decay)


        self.models_label = [f"model {i}" for i in range(len(models))]
        self.models = models
        self.models_loss = [] 
        self.models_acc = [] 

        for model, lr, wd  in zip(self.models, learning_rate, weight_decay):
            
            self.train_model(model, learning_rate=lr, weight_decay=wd, batch_size=batch_size, max_epochs=max_epochs )

            self.models_loss.append(self.losses)
            self.models_acc.append(self.accuracies)


    def plotAcc(self):
        """ Plot validation accuracies to determine best model """
        plt.figure(figsize=(8, 5))
        for lab, accs in zip(self.models_label, self.models_acc):
            plt.plot(self.epochs, accs, label=[lab+", train", lab+", val"])
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")


    def plotLosses(self):
        """ Plot validation accuracies to determine best model """
        plt.figure(figsize=(8, 5))
        plt.title("Training Loss")
        models_loss = np.array(self.models_loss)
        plt.plot(self.epochs, models_loss.T, label=self.models_label)
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        
        
    def bestModelAcc(self):
        """
        Prints test accuracy of best model
        Chooses model that yields the best validation accuracy
        S is object containing the data used during training 
        """
        best_acc_idx = np.argmax([acc[-1, -1] for acc in self.models_acc])
        best_model = self.models[best_acc_idx]
        best_acc = self.acc_te(best_model)
        print(f"Accuracy of test set of best model (idx={best_acc_idx}): {best_acc*100:.1f}%")


def resample_with_replacement(X, y, no_combinations, n_channels):
    """Make samples for a given dataset containing multiple users."""

    newX = []
    newy = []
    for u in np.unique(y):    # Loop over all users 

        Xuser = X[y==u]    
        Xcomb = get_combinations(Xuser, n_channels, no_combinations)

        newX.append(Xcomb)
        newy.append(np.full(no_combinations, u))

    return np.concatenate(newX), np.concatenate(newy)


def get_combinations(X, n_channels, no_combinations):
    """From X.shape[0] choose n_channels, repeated no_combinations times."""

    result = np.zeros((no_combinations, n_channels, *X.shape[1:]))
    for i in range(no_combinations):
        result[i] = X[np.random.randint(0, X.shape[0], size=n_channels)] 
    return result


def load_data(dataPath, triggers=True, releases=False):

    assert (triggers or releases), "At least one of triggers or releases need to be set to True!"
    data = np.load(dataPath)

    # Check different users
    users = np.unique(np.array([key.split("_")[0] for key in data], dtype=str))

    # Build X data and corresponding labels
    Xraw = []
    yraw = []

    # def appendData(key):
    #     Xraw.append(data[key])
    #     yraw.append(np.full(len(data[key]), np.argwhere(users==key.split("_")[0])[0]))
    #
    # for key in data:
    #     _, mode = key.split("_")
    #
    #     if triggers and (mode=="triggers"): appendData(key)
    #     if releases and (mode=="releases"): appendData(key)

    # TODO: Need to find a better way to create Xraw with shape (N, 2, 32) or (N, 1, 32)
    # Depending on using just triggers or triggers+releases

    for u in users:

        if triggers:
            Xraw.append(data[u+"_triggers"])
            yraw.append(np.full(len(data[u+"_triggers"]), np.argwhere(users==u)[0]))

        if releases:
            Xraw.append(data[u+"_releases"])
            yraw.append(np.full(len(data[u+"_releases"]), np.argwhere(users==u)[0]))


    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw

