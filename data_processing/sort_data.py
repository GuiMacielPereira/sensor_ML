
# A script to take the data from csv files and put them in a single npz files

from pathlib import Path
import pandas as pd
import numpy as np


currentPath = Path(__file__).absolute().parent   # Get path of folder
rawDataPath = currentPath / "raw_data"

C1 = []
C2 = []
for f in rawDataPath.iterdir():
    if f.is_file:
        d = pd.read_csv(f).iloc[:, 1]
        C1.append(d) if f.name[0] == 'g' else C2.append(d)

C1 = np.array(C1)
C2 = np.array(C2)

print("Loaded data: ", C1.shape, C2.shape)
print("First voltages:\n", C1[:, :5])

savePath = currentPath / "sorted_data.npz"
np.savez(savePath, C1=C1, C2=C2)

