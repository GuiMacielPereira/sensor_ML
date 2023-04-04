
# Script to merge data files
from peratouch.config import data_dir 
import numpy as np
import pandas as pd

data_to_concat = []
for i in range(1, 11):
    path = data_dir / "raw_csv" / "five_users"/ f"run{i}_data.csv" 

    data = pd.read_csv(str(path)).to_numpy()
    data = np.roll(data, shift=+(i-1), axis=1)
    data_to_concat.append(data.T)

merged_data = np.concatenate(data_to_concat, axis=1)
# Shape (5, n_points)
# Order of Rows: [G, E, LJ, A, J]

dict_to_save = {}
for i, sig in enumerate(merged_data):
    dict_to_save["U"+str(i)] = sig

save_path = data_dir / "raw_npz" / "five_users_data.npz"
np.savez(save_path, **dict_to_save)

for k in dict_to_save: print(f"{k} : {dict_to_save[k].shape}")
