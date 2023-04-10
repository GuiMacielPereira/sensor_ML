# %%
# Notebook for processing raw data 
# Selects only triggers and releases
# Load data
import numpy as np
import matplotlib.pyplot as plt
from peratouch.config import data_dir
from peratouch.preprocessing import TriggersAndReleases
# Width of triggers 
width = 32 
path = data_dir / "raw_npz" / "five_users_data.npz"
data = np.load((path))
# %%
# Look at a few triggers of one of the users
key = list(data.keys())[0] 
TR = TriggersAndReleases(data)
TR.plot_signal(key)
TR.run()
#%%
TR.plot_clean(key)
# %%
TR.plot_discarded(key)
# %%
# Compare the mean and std of the signals for all the users involved
TR.plot_means_std()
#%%
# Save file
filename = path.name.split(".")[0] + "_window" + f"_{TR.get_triggers()[key].shape[1]}.npz"
save_path = data_dir / "processed" / filename
TR.save_dict(save_path)


