import numpy as np
import pandas as pd


data_to_concat = []
for i in range(1, 11):
    data = pd.read_csv(f"./run{i}_data.csv").to_numpy()
    data = np.roll(data, shift=+(i-1), axis=1)
    data_to_concat.append(data.T)

merged_data = np.concatenate(data_to_concat, axis=1)
print("Data to save shape: ", merged_data.shape)
# Order of Rows: [G, E, LJ, A, J]

dict_to_save = {}
for i, sig in enumerate(merged_data):
    dict_to_save["U"+str(i)] = sig

np.savez("./five_people.npz", **dict_to_save)



