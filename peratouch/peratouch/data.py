import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import partial
from imblearn.over_sampling import RandomOverSampler

class Data:
    def __init__(self, dataPath, triggers=True, releases=False, transforms=False):
        self.Xraw, self.yraw = load_data(dataPath, triggers, releases, transforms)



    def group_presses(self, n_elements=3):

        def group(x):
            res = []
            for i in range(len(x) - n_elements):
                res.append(x[i:i+n_elements, 0, :])
            return np.array(res)

        self.Xraw, self.yraw = act_on_user(group, self.Xraw, self.yraw)

    # def shuffle(self):   # Shuffle presses randomly
    #     self.Xraw, self.yraw = sklearn.utils.shuffle(self.Xraw, self.yraw, random_state=42)

    def split(self):
        self.Xtrain, Xtest, self.ytrain, ytest = train_test_split(self.Xraw, self.yraw, test_size=0.20, random_state=42)
        self.Xtest, self.Xval, self.ytest, self.yval = train_test_split(Xtest, ytest, test_size=0.50, random_state=42)

    def balance_train(self): 
        # For some weird reason, resampler takes only up to 2 dims, so need to do some reshaping tricks
        batch_size, n_ch, in_size = self.Xtrain.shape
        Xtrain = self.Xtrain.reshape(batch_size, -1)
        Xtrain, self.ytrain = RandomOverSampler(random_state=42).fit_resample(Xtrain, self.ytrain)
        self.Xtrain = Xtrain.reshape(-1, n_ch, in_size)

    # # NOTE: Tried this function, much worse accuracy
    # # Destroyes consecutive presses by random sampling
    # def shuffle_presses_train(self):
    #
    #     def shuffle(x):
    #         batch_size, _, input_size = x.shape
    #         x = x.reshape(-1, input_size)
    #         x = sklearn.utils.shuffle(x)
    #         return x.reshape(batch_size, -1, input_size)
    #
    #     self.Xtrain, self.ytrain = act_on_user(shuffle, self.Xtrain, self.ytrain)

    def normalize(self, verbose=True):
        """Normalise datasets according to fixed value from train set"""
        # Fix normalisation value
        xmax = np.mean(np.max(self.Xtrain, axis=-1, keepdims=True), axis=0, keepdims=True)     # Hard coding the normalization severely affects validation accuracy

        self.Xtrain /= xmax
        self.Xtest /= xmax
        self.Xval /= xmax
        
        if verbose:
            print("Train, test and validation data normalized to:")
            for x in (self.Xtrain, self.Xtest, self.Xval):
                print(f"{np.mean(np.max(x, axis=-1), axis=0)}")

    def reshape_for_lstm(self, input_size, sliding=False):
        
        def reshape(x):
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])     # Concatenate several triggers together, if multiple triggers present
            if sliding:
                res = []
                for i in range(x.shape[-1] - input_size + 1):
                    res.append(x[:, i:i+input_size])
                return np.concatenate(res, axis=1)
            else:
                if x.shape[-1] % input_size: raise ValueError("Splitting size not matching!")
                return x.reshape((x.shape[0], -1, input_size))

        self.Xtrain = reshape(self.Xtrain) 
        self.Xtest = reshape(self.Xtest) 
        self.Xval = reshape(self.Xval) 

    # NOTE: This method creates groups of 3 triggers by random resampling
    # Acts on each dataset separately, so test data is still completely separate
    # This provides more information per multiple triggers, and so achieves much higher accuracies
    # def resample_datasets(self):
    #
    #     np.random.seed(0)
    #     def make_combinations(X):
    #         return resample_with_replacement(X, n_channels=3, no_combinations=len(X))
    #
    #     self.Xtrain, self.ytrain = act_on_user(make_combinations, self.Xtrain, self.ytrain)
    #     self.Xtest, self.ytest = act_on_user(make_combinations, self.Xtest, self.ytest)
    #     self.Xval, self.yval = act_on_user(make_combinations, self.Xval, self.yval)


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
            yt = torch.tensor(y, dtype=torch.long).to(self.device)    # Use for CrossEntropyLoss
            return xt, yt 
        
        self.xtr, self.ytr = to_tensor(self.Xtrain, self.ytrain)
        self.xv, self.yv = to_tensor(self.Xval, self.yval)
        self.xte, self.yte = to_tensor(self.Xtest, self.ytest)

        # Create trainset in the correct format for dataloader
        self.trainset = [[x, y] for (x, y) in zip(self.xtr, self.ytr)]

    def print_shapes(self):
        print(
            "\nRaw data shape: ", self.Xraw.shape, \
            "\nLabels shape: ", self.yraw.shape,  \
            "\nShape of test set:", self.Xtest.shape,  \
            "\nShape of train set:", self.Xtrain.shape,  \
            "\nShape of validation set:", self.Xval.shape, \
            "\nUnique labels: ", np.unique(self.yraw),  \
            "\nFraction of test labels: ", [np.round(np.mean(self.ytest==i), 2) for i in np.unique(self.ytest)], \
            "\nFraction of validation labels: ", [np.round(np.mean(self.yval==i), 2) for i in np.unique(self.yval)], \
            "\nFraction of train labels: ", [np.round(np.mean(self.ytrain==i), 2) for i in np.unique(self.ytrain)], \
            "\ndtype of inputs: ", self.xtr.dtype
            )

    def acc_tr(self, model):
        return acc(model, self.xtr, self.ytr)

    def acc_val(self, model):
        return acc(model, self.xv, self.yv)

    def acc_te(self, model):
        return acc(model, self.xte, self.yte)

    def matthews_corrcoef_te(self, model):
        return matthews_corrcoef(model, self.xte, self.yte)

    def loss_val(self, model, criterion):
        with torch.no_grad():    # Each time model is called, need to avoid updating the weights
            return criterion(model(self.xv), self.yv).item()
 
    def loss_tr(self, model, criterion):
        with torch.no_grad():    # Each time model is called, need to avoid updating the weights
            return criterion(model(self.xtr), self.ytr).item()

def acc(model, x, y):
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out.data, 1)
        return (pred==y).detach().cpu().numpy().mean()

def matthews_corrcoef(model, x, y):
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out.data, 1)
        pred = pred.detach().cpu().numpy()
        return sklearn.metrics.matthews_corrcoef(y, pred)

def test_metrics(data_objects, models):
    print("\nTest dataset metrics:")
    for i, (D, model) in enumerate(zip(data_objects, models)):
        print(f"Model {i}: Accuracy = {D.acc_te(model)*100:.1f}%, Matthews Corr Coef = {D.matthews_corrcoef_te(model):.2f}")

# Data Functions 
def act_on_user(func, X, y):
    """
    Takes in a given function for grouping or resampling 
    and applies it to individual users, creating corresponding label.
    """
    newX = []
    newy = []
    for u in np.unique(y):    # Loop over all users

        Xuser = X[y==u]   
        Xtrans = func(Xuser)

        newX.append(Xtrans)             # Store new shape 
        newy.append(np.full(len(Xtrans), u))   # Make new label

    return np.concatenate(newX), np.concatenate(newy)


# def resample_with_replacement(X, n_channels, no_combinations):
#     """From X.shape[0] choose n_channels, repeated no_combinations times."""
#
#     result = np.zeros((no_combinations, n_channels, *X.shape[1:]))
#     for i in range(no_combinations):
#         result[i] = X[np.random.randint(0, X.shape[0], size=n_channels)]   # index 0 to match shape
#
#     # Reshape into correct number of channels.
#     # Accounts for case where both triggers and releases are considered.
#     return result.reshape((no_combinations, n_channels*X.shape[1], *X.shape[2:]))


# Loading function 
def load_data(dataPath, triggers=True, releases=False, transforms=False):

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


