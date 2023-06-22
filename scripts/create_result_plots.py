
from peratouch.config import path_analysis_results, path_results, path_figures
from peratouch.plot import plot_dataset_sizes, plot_presses_users, plot_lstm_sizes
import datetime
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

day = datetime.date.today()
day = '2023-05-17'

# load_path = path_analysis_results / f'input_size_lstm_2023-05-06.npz'
# n_ch = 1
# n_epochs = 100
# n_folds  = 5
# n_batches = 15
# plot_lstm_sizes(load_path)
#
# plt.show()
# exit()
#
load_path = path_analysis_results / f'dataset_size_2023-05-04.npz'
plot_dataset_sizes(load_path)
plt.show()
#

# load_path = path_analysis_results / f'no_presses_users_{day}.npz'
# load_path = path_analysis_results / 'no_presses_users_2023-05-18.npz'
#
# data = np.load(load_path)
# for k in data: print(len(data[k][-1]))

# plot_presses_users(load_path)
# plt.show()

# path_results = path_results / 'n_press_3_resampled' 
#
# for m in ['CNN', 'LSTM', 'CNN_LSTM']:
#     print(f'\nResults of {m}:')
#     data = np.load(str(path_results / f'{m}_preds_{day}.npz'))
#     print(metrics.classification_report(data['actual_vals'], data['predictions'], digits=3))
#     with sns.axes_style('dark'):
#         metrics.ConfusionMatrixDisplay.from_predictions(data['actual_vals'], data['predictions'], cmap='plasma')
#         plt.savefig(str(path_results / f'{m}_conf_{day}.pdf'), bbox_inches='tight')
# plt.show()
#

