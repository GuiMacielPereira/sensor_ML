import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from peratouch.plot import plot_X
import sklearn
from sklearn.model_selection import train_test_split

class Data:

    def __init__(self, Xraw, yraw):
        self.Xraw = Xraw
        self.yraw = yraw

    def group_presses(self, n_press=3):

        if n_press==1:
            return     # Skip routine if not grouping presses

        def group(x):
            res = []
            for i in range(0, len(x) - n_press, n_press):
                res.append(x[i:i+n_press, 0, :])
            return np.array(res)

        self.Xraw, self.yraw = act_on_user(group, self.Xraw, self.yraw)

    def shuffle(self):   # Shuffle presses randomly
        self.Xraw, self.yraw = sklearn.utils.shuffle(self.Xraw, self.yraw, random_state=42)

    def make_folds(self, n_folds):
        """Splits dataset into folds without shuffling."""
        if n_folds < 2:
            raise ValueError("Need to have at least two folds (50% test size). To avoid running all folds in cross validation, use n_runs parameter.")

        kf = sklearn.model_selection.KFold(n_splits=n_folds) 
        self.folds_idxs = kf.split(self.Xraw)

    def next_fold(self):
        """Selects next fold from pre-determined folds"""
        print("\n\n-- New Fold --")
        train_idx, test_idx = next(self.folds_idxs)

        # Spliting below ensures test folds coves entirety of dataset
        self.Xtest, self.ytest = self.Xraw[test_idx], self.yraw[test_idx]
        # Now split train fold into train and validation sets
        self.Xtrain, self.Xval, self.ytrain, self.yval = train_test_split(self.Xraw[train_idx], self.yraw[train_idx], test_size=0.15, shuffle=False)

    def balance_train(self): 
        # For some weird reason, resampler takes only up to 2 dims, so need to do some reshaping tricks
        batch_size, n_ch, in_size = self.Xtrain.shape
        Xtrain = self.Xtrain.reshape(batch_size, -1)
        Xtrain, self.ytrain = RandomOverSampler(random_state=42).fit_resample(Xtrain, self.ytrain)
        self.Xtrain = Xtrain.reshape(-1, n_ch, in_size)

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
    def resample_presses(self, n_press=3):

        if n_press==1:
            return

        np.random.seed(0)
        def make_combinations(X):
            return resample_with_replacement(X, n_channels=n_press, no_combinations=len(X))

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
def load_data(dataPath):
    data = np.load(dataPath)

    # Check different users
    users = np.unique(np.array([key.split("_")[0] for key in data], dtype=str))

    # Build X data and corresponding labels
    Xraw = []
    yraw = []
    for u in users:

        userX = []

        userX.append(data[u+"_triggers"])
        
        Xraw.append(np.stack(userX, axis=1))
        yraw.append(np.full(len(userX[0]), np.argwhere(users==u)[0]))

    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw


