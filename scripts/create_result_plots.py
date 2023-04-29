
from peratouch.config import path_analysis_figures
from peratouch.config import path_analysis_results
from peratouch.plot import plot_dataset_sizes, plot_presses_users
import datetime
import matplotlib.pyplot as plt

day = datetime.date.today()

load_path = path_analysis_results / f'dataset_size_{day}.npz'
plot_dataset_sizes(load_path)
plt.savefig(str(path_analysis_figures / f'dataset_size_{day}.pdf'))
plt.show()

load_path = path_analysis_results / f'no_presses_users_{day}.npz'
plot_presses_users(load_path)
plt.savefig(str(path_analysis_figures / f'no_presses_users_{day}.pdf'))

plt.show()
