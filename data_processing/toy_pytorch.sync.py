#%%
# This mounts your Google Drive to the Colab VM.
# from google.colab import drive
# drive.mount('/content/drive')
#
# # TODO: Enter the foldername in your Drive where you have saved the unzipped
# # assignment folder, e.g. 'cs231n/assignments/assignment2/'
# FOLDERNAME = 'Masters_Project/data_processing'
# assert FOLDERNAME is not None, "[!] Enter the foldername."
#
# # Now that we've mounted your Drive, this ensures that
# # the Python interpreter of the Colab VM can load
# # python files from within it.
# import sys
# sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))
#

#%%
# Load triggers data

import numpy as np 
from sklearn.model_selection import train_test_split

triggers = np.load("./triggers_data.npz")

Xraw = []
yraw = []
for i, key in enumerate(triggers):
    sig = triggers[key]
    Xraw.append(sig)
    yraw.append(np.full(len(sig), i))

Xraw = np.concatenate(Xraw)
yraw = np.concatenate(yraw)
Xraw.shape

#%% 
# Split data into test  and train set
Xtrain, Xtest, ytrain, ytest = train_test_split(Xraw, yraw, test_size=0.1, random_state=42)
print("Fraction of single class in test set: ", np.mean(ytest==0))
print("Size of test set:", Xtest.shape)

# Split train data into train and validation set
Xfold, Xval, yfold, yval = train_test_split(Xtrain, ytrain, test_size=0.1, random_state=42)
print("Size of train set:", Xtrain.shape)


#%%
# Function to print accuracy of validation set while training
def checkAcc(x, y, model, set="Validation"):
    scores = model(x)
    _, preds = scores.max(1)    # Returns indices alongside axis=1
    num_correct = (preds == y).sum()
    num_samples = preds.size(0)
    acc = float(num_correct) / num_samples
    print(f"{set} Accuracy: {num_correct} / {num_samples} = {acc:.2f}")


#%%
# Try to code a simple network
import torch

device = torch.device('cpu')
dtype = torch.float32

N, D_in, H, D_out = len(Xtrain), 30, 1000, 2 

# Create random Tensors to hold inputs and outputs
x = torch.tensor(Xfold, device=device, dtype=dtype)
y = torch.tensor(yfold, device=device, dtype=torch.long)

xv = torch.tensor(Xval, device=device, dtype=dtype)
yv = torch.tensor(yval, device=device, dtype=torch.long)

#%%
# Define a general procedure to train a model

def trainModel(model, loss_fn, optimizer, x, y, xv, yv, N=1000):
    for t in range(N):

        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # Internally, this call will compute gradients for all learnable parameters in the model.
        loss.backward()

        # Update weights
        optimizer.step()
        # Zero the gradients after weight updates.
        optimizer.zero_grad()

        with torch.no_grad():     # Very important to specify no_grad to avoid automatic differentiation of this step
            if t%100==0:
                print(f"Iteration {t}\nLoss: {loss.item()}")
                checkAcc(xv, yv, model)

#%% 
# Create a simple architecture first
# Use nn.Sequention API 
# Each layer has its own internal weights and bias 
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        ).to(device)

# Loss function to use - check available functions on pytorch
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# trainModel(model, loss_fn, optimizer, x, y, xv, yv)

#%%
# Test model on test data
xt= torch.tensor(Xtest, device=device, dtype=dtype)
yt= torch.tensor(ytest, device=device, dtype=torch.long)
# with torch.no_grad():
#     checkAcc(xt, yt, model, set="Test")

#%%
# Normalizing the data and building a deeper layer model
def normalize(x, xtr):
    # Use quantities from training data set
    # avgmax = np.mean(np.max(xtr, axis=1))
    avgmax = np.max(x, axis=1)[:, np.newaxis]
    return x / avgmax 

xdata = [Xfold, Xval, Xtest]
for i in range(len(xdata)):
    xdata[i] = normalize(xdata[i], Xfold)[:, np.newaxis, :]

# Pass onto pytorch tensors
for i in range(len(xdata)):
    xdata[i] = torch.tensor(xdata[i], device=device, dtype=dtype)

xnorm, xvnorm, xtnorm = xdata
print(xnorm.shape)

class deepModel(torch.nn.Module):
    def __init__(self):
        super(deepModel, self).__init__()

        D1 = 12 
        D2 = 12 

        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv1d(1, D1, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2)
                )
        
        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv1d(D1, D2, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=3)
                )
        # self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(D2 * 5, 100)    
        self.fc2 = torch.nn.Linear(100, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)   # Flatten tensor
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model_deep = deepModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optim_deep = torch.optim.SGD(model_deep.parameters(), lr=1e-3, momentum=0.9)

trainModel(model_deep, loss_fn, optim_deep, xnorm, y, xvnorm, yv, 10000)
with torch.no_grad():
    checkAcc(xtnorm, yt, model_deep, set="Test")


#%%
import matplotlib.pyplot as plt

def plotTest(Xtest, ytest, predictions):
    usr1 = predictions==0
    usr2 = ~usr1
    for j, usr in enumerate([usr1, usr2]):
        X1 = Xtest[usr]
        labels = ytest[usr]
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Predictions for user {j}")
        for i, (x, lab) in enumerate(zip(X1, labels)):
            plt.subplot(1, len(X1), i+1)
            plt.plot(range(len(x)), x, "b.")
            plt.title(f"Real User: {lab}")

# Plot results
with torch.no_grad():     # Very important to specify no_grad to avoid automatic differentiation of this step
    scores = model_deep(xtnorm).view(xtnorm.shape[0], -1)
    _, preds = scores.max(1)
    predictions = preds.detach().numpy()

plotTest(Xtest, ytest, predictions)
