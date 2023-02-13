# Script to store all of the main functions for cleaning and loading data
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

class SensorSignals:

    def __init__(self, dataPath):
        self.Xraw, self.yraw = load_data(dataPath)

    def split_data(self):
        Xtrain, self.Xtest, ytrain, self.ytest = train_test_split(self.Xraw, self.yraw, test_size=0.15, random_state=42)
        self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(Xtrain, ytrain, test_size=0.15, random_state=42)

    def norm_X(self):

        def norm(x):
            # Hard code a value based on the training set
            # xmax = np.mean(np.max(Xtrain, axis=1))     # Hard coding the normalization severely affects validation accuracy
            xmax = np.max(x, axis=1)[:, np.newaxis]
            return x / xmax 

        self.Xtrain = norm(self.Xtrain)
        self.Xtest = norm(self.Xtest)
        self.Xval = norm(self.Xval)
        print("\nTrain, Test and Validation set were normalized!")

    def setup_tensors(self):
        # Use GPU if available 
        self.device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
        self.dtype = torch.float32
        print("Using Device: ", self.device, ", dtype: ", self.dtype)

        def toTensor(X, y):
            xt = torch.tensor(X[:, np.newaxis, :], dtype=self.dtype)
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
    


def load_data(dataPath, triggers=True, releases=False):
    data = np.load(dataPath)

    # Check different users
    users = np.unique(np.array([key.split("_")[0] for key in data], dtype=str))

    # Build X data and corresponding labels
    Xraw = []
    yraw = []

    def appendData(key):
        Xraw.append(data[key])
        yraw.append(np.full(len(data[key]), np.argwhere(users==key.split("_")[0])[0]))

    for key in data:
        _, mode = key.split("_")

        if triggers and (mode=="triggers"): appendData(key)
        if releases and (mode=="releases"): appendData(key)

    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw


