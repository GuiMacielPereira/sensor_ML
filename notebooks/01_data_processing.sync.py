# %%
# Notebook for processing raw data 
# Selects only triggers and releases

# Load data
import numpy as np
import matplotlib.pyplot as plt
from peratouch.config import data_dir
from peratouch.data import TriggersAndReleases

# Width of triggers 
width = 32 
path = data_dir / "raw_npz" / "five_users_data.npz"
data = np.load((path))

# %%
# Look at a few triggers of one of the users
key = list(data.keys())[0] 
signal = data[key]   # Only a few presses 

TR = TriggersAndReleases(signal)
TR.plot_signal()
TR.run()
TR.plot_clean()
# %%
TR.plot_noisy()
TR.plot_short()
# %%
# Compare the mean and std of the signals for all the users involved
plt.figure(figsize=(13, 7))
n_users = len(data)
for i, key in enumerate(data):
    TR = TriggersAndReleases(data[key])
    TR.run()
    for j, sig in enumerate([TR.get_triggers(), TR.get_releases()]):
        plt.subplot(n_users, 2, 2*i+j+1)
        sigAvg = np.mean(sig, axis=0)
        sigStd = np.std(sig, axis=0)
        plt.title(key+" mean+std")
        plt.errorbar(np.arange(len(sigAvg)), sigAvg, sigStd, fmt="b.")
        plt.xticks([])

# %%
# Loop to build and save data 
# Only run if want to update saved file 
processedData = {}
for key in data:
    print(f"\nUser {key}:")
    TR = TriggersAndReleases(data[key])
    TR.run()

    processedData[key+"_triggers"] = TR.get_triggers()
    processedData[key+"_releases"] = TR.get_releases()

filename = path.name.split(".")[0] + "_window" + f"_{TR.get_triggers().shape[1]}.npz"
save_path = data_dir / "processed" / filename
np.savez(save_path, **processedData)
print("\n\n")
for k in processedData: print(f"Saved {k} : {processedData[k].shape}")


