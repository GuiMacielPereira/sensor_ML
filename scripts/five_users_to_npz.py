
# Script to merge data files
# In the case of five users, each column in the csv file corresponds to a single sensor
# The initial order of users: [G, E, LJ, A, J] 
# The users rotated (equivalent to roll) each subsquent song
from peratouch.config import data_dir 
import numpy as np
import pandas as pd

folder = "main_collection"

data_to_concat = []
i = 1
# for i in range(1, 11):
while True:

    path = data_dir / "raw_csv" / "five_users"/ folder / f"run{i}.csv" 
    try:
        data = pd.read_csv(str(path)).to_numpy()
    except FileNotFoundError:
        break
    data = np.roll(data, shift=+(i-1), axis=1)
    data_to_concat.append(data.T)
    print(f"Read file run{i}.csv")
    i+=1

merged_data = np.concatenate(data_to_concat, axis=1)
# Shape (5, n_points)
# Order of rows for first collection: [G, E, LJ, A, J]
# Order of rows for main collection: [G, M, LJ, P, J]

dict_to_save = {}
for i, sig in enumerate(merged_data):
    dict_to_save["U"+str(i)] = sig

save_path = data_dir / "raw_npz" / f"five_users_{folder}.npz"
np.savez(save_path, **dict_to_save)

for k in dict_to_save: print(f"{k} : {dict_to_save[k].shape}")
