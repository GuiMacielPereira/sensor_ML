
# Script to merge data files
# In the case of five users, each column in the csv file corresponds to a single sensor
# The initial order of users: [G, E, LJ, A, J] 
# The users rotated (equivalent to roll) each subsquent song
from peratouch.config import data_dir 
import numpy as np
import pandas as pd

data_to_concat = []
for i in range(1, 31):     # Excludes run 31, corresponding to fire and flames
    path = data_dir / "raw_csv" / f"run{i}.csv" 
    data = pd.read_csv(str(path)).to_numpy()
    data = np.roll(data, shift=+(i-1), axis=1)
    data_to_concat.append(data.T)
    print(f"Read file run{i}.csv")

merged_data = np.concatenate(data_to_concat, axis=1)
# Shape (5, n_points)
# Order of rows during collection: [G, M, LJ, P, J]

dict_to_save = {}
for i, sig in enumerate(merged_data):
    dict_to_save["U"+str(i)] = sig

save_path = data_dir / "raw_npz" / f"five_users.npz"
np.savez(save_path, **dict_to_save)

for k in dict_to_save: print(f"{k} : {dict_to_save[k].shape}")
