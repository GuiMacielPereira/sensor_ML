
# A script to take the data from csv files and put them in single npz files
# Iterates over all files and combines those that start with the same initial. 
# The initials to look for are provided in form of a list 
# eg. ['A', 'G']
from peratouch.config import data_dir
import pandas as pd
import numpy as np

# Create dictionary to store merge results for each user
initialsToMerge =  ['A', 'G', 'J']
data = {}
for c in initialsToMerge:
    data[c] = []

# Read data onto dictionary
path = data_dir / "raw_csv" / "three_users"
for f in path.glob('**/*.csv'):
    data[f.name[0]].append(pd.read_csv(f).iloc[:, -1]) # Read only column corresponding to values

# Pass lists into arrays
for key in data: data[key] = np.concatenate(data[key])

save_path = data_dir / "raw_npz" / "three_users_data.npz"
np.savez(save_path, **data)

# Print shape of saved arrays 
for k in data: print(f"{k} : {data[k].shape}")

