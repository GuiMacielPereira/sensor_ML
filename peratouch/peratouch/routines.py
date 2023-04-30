
from peratouch.data import Data, load_data
from peratouch.trainer import Trainer 
from peratouch.results import Results 
# from peratouch.networks import CNN
# import seaborn as sns
import sklearn
import numpy as np

def run_network(network, Xraw, yraw, n_ch=1, n_epochs=20, n_folds=5, n_runs=5,  
        plots=True, n_batches=20, random_resampling=False, input_size=4, hidden_size=8):
    """
    This humoungous and complicated funciton defines the routine based on input parameters.
    """

    n_out = len(np.unique(yraw))

    D = Data(Xraw, yraw)

    if (n_ch>1) & ~random_resampling:
        D.group_presses(n_press=n_ch)

    D.shuffle()      # Shuffling destroys order 

    # Create indices of several folds
    D.make_folds(n_folds)     # Makes indices available inside class

    predictions = []
    actual_vals = []

    for _ in range(n_runs):
        D.next_fold()         # Selects train, validation and test set
        D.normalize()

        if (n_ch>1) & random_resampling:
            D.resample_presses(n_press=n_ch)

        if network.__name__ == 'LSTM':
            D.reshape_for_lstm(input_size=input_size)

        D.tensors_to_device()
        D.print_shapes()

        if plots:
            D.plot_data()

        if network.__name__ == 'LSTM':
            model = network(input_size=input_size, hidden_size=hidden_size)
        else:
            model = network(n_ch=n_ch, out_size=n_out)      # Initialize new model each fold

        T = Trainer(D)
        T.setup(model, max_epochs=n_epochs, batch_size=int(len(D.xtr)/n_batches))       # 20 minibatches
        T.train_model(model)

        if plots:
            T.plot_train()

        R = Results(D, model)
        preds, actual = R.get_preds_actual()
        R.test_metrics()

        predictions.extend(preds)
        actual_vals.extend(actual)
    
    if plots:
        print(sklearn.metrics.classification_report(actual_vals, predictions))
        # TODO: Solve error from running below
        # with sns.axes_style('dark'):
        #     sklearn.metrics.ConfusionMatrixDisplay.from_predictions(actual_vals, predictions)

    return actual_vals, predictions 

