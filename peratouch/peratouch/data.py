import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from peratouch.plot import plot_X
import sklearn

class Data:
    def __init__(self, dataPath, triggers=True, releases=False):
        self.Xraw, self.yraw = load_data(dataPath, triggers, releases)

    def group_presses(self, n_elements=3):

        def group(x):
            res = []
            for i in range(len(x) - n_elements):
                res.append(x[i:i+n_elements, 0, :])
            return np.array(res)

        self.Xraw, self.yraw = act_on_user(group, self.Xraw, self.yraw)

    # ------ New functions to run Cross Validation
    def shuffle(self):   # Shuffle presses randomly
        self.Xraw, self.yraw = sklearn.utils.shuffle(self.Xraw, self.yraw, random_state=42)

    def halve_raw_data(self):
        self.Xraw, _ = np.array_split(self.Xraw, 2)
        self.yraw, _ = np.array_split(self.yraw, 2)

    def make_folds(self, n_folds=5):
        kf = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.folds_idxs = kf.split(self.Xraw)

    def next_fold(self):
        train_idx, test_idx = next(self.folds_idxs)
        test_idx, val_idx = np.array_split(test_idx, 2)     # Does not raise error if not equal split
        self.Xtrain, self.Xtest, self.Xval = self.Xraw[train_idx], self.Xraw[test_idx], self.Xraw[val_idx]
        self.ytrain, self.ytest, self.yval = self.yraw[train_idx], self.yraw[test_idx], self.yraw[val_idx]
    # -------------------

    # def split(self):
    #     self.Xtrain, Xtest, self.ytrain, ytest = train_test_split(self.Xraw, self.yraw, test_size=0.20, random_state=42)
    #     self.Xtest, self.Xval, self.ytest, self.yval = train_test_split(Xtest, ytest, test_size=0.50, random_state=42)

    # # NOTE: Dirty function to halve all datasets, used only to see how size of datasets affects accuracy
    # def halve_datasets(self):
    #     self.Xtrain  = self.Xtrain[:int(len(self.Xtrain)/2)]
    #     self.Xtest = self.Xtest[:int(len(self.Xtest)/2)]
    #     self.Xval = self.Xval[:int(len(self.Xval)/2)]
    #     self.ytrain = self.ytrain[:int(len(self.ytrain)/2)]
    #     self.ytest = self.ytest[:int(len(self.ytest)/2)]
    #     self.yval = self.yval[:int(len(self.yval)/2)]

    def balance_train(self): 
        # For some weird reason, resampler takes only up to 2 dims, so need to do some reshaping tricks
        batch_size, n_ch, in_size = self.Xtrain.shape
        Xtrain = self.Xtrain.reshape(batch_size, -1)
        Xtrain, self.ytrain = RandomOverSampler(random_state=42).fit_resample(Xtrain, self.ytrain)
        self.Xtrain = Xtrain.reshape(-1, n_ch, in_size)

    # # NOTE: Tried this function, much worse accuracy
    # # Destroyes consecutive presses by random sampling on train dataset
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
            norm = lambda x: np.mean(np.max(x, axis=-1), axis=0) 
            print("Train, test and validation arrays normalized to:")
            with np.printoptions(precision=4):
                print(f"{norm(self.Xtrain)}, {norm(self.Xtest)}, {norm(self.Xval)}")

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

    # NOTE: This method creates groups of 3 triggers by random resampling over entire dataset
    # Acts on each dataset separately, so test data is still completely separate
    # This provides more information per multiple triggers, and so achieves much higher accuracies
    def resample_triggers(self, n_channels=3):
        np.random.seed(0)
        def make_combinations(X):
            return resample_with_replacement(X, n_channels=n_channels, no_combinations=len(X))

        self.Xtrain, self.ytrain = act_on_user(make_combinations, self.Xtrain, self.ytrain)
        self.Xtest, self.ytest = act_on_user(make_combinations, self.Xtest, self.ytest)
        self.Xval, self.yval = act_on_user(make_combinations, self.Xval, self.yval)

    def plot_data(self):
        plot_X(self.Xtrain, self.ytrain)

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


# NOTE: Function used to resample over entire dataset
def resample_with_replacement(X, n_channels, no_combinations):
    """From X.shape[0] choose n_channels, repeated no_combinations times."""

    result = np.zeros((no_combinations, n_channels, *X.shape[1:]))
    for i in range(no_combinations):
        result[i] = X[np.random.randint(0, X.shape[0], size=n_channels)]   # index 0 to match shape

    # Reshape into correct number of channels.
    # Accounts for case where both triggers and releases are considered.
    return result.reshape((no_combinations, n_channels*X.shape[1], *X.shape[2:]))


# Loading function 
def load_data(dataPath, triggers=True, releases=False):
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
            # NOTE: Some transforms were tried at this point but were abandoned.
        if releases:
            userX.append(data[u+"_releases"])
        
        Xraw.append(np.stack(userX, axis=1))
        yraw.append(np.full(len(userX[0]), np.argwhere(users==u)[0]))

    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw


