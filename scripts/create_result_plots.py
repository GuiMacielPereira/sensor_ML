
from peratouch.config import path_analysis_results, path_results, path_figures
from peratouch.plot import plot_dataset_sizes, plot_presses_users
import datetime
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

day = datetime.date.today()

load_path = path_analysis_results / f'dataset_size_{day}.npz'
plot_dataset_sizes(load_path)
#
# load_path = path_analysis_results / f'no_presses_users_{day}.npz'
# plot_presses_users(load_path)
plt.show()

exit()

for m in ['CNN', 'LSTM', 'CNN_LSTM']:
    print(f'\nResults of {m}:')
    data = np.load(str(path_results / f'{m}_preds_{day}.npz'))
    print(metrics.classification_report(data['actual_vals'], data['predictions'], digits=3))
    with sns.axes_style('dark'):
        metrics.ConfusionMatrixDisplay.from_predictions(data['actual_vals'], data['predictions'], cmap='plasma')
        plt.savefig(str(path_figures / f'{m}_conf_{day}.pdf'), bbox_inches='tight')
plt.show()


