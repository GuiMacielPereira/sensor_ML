
# A script to take the data from csv files and put them in single npz files
from pathlib import Path
import pandas as pd
import numpy as np
currentPath = Path(__file__).absolute().parent   # Get path of folder


def mergeFiles(dataFolder:Path, initialsToMerge:list):
    """
    Iterates over all files in dataFolder path and 
    combines together files that start with the same initial. 
    The initials to look for are provided in form of a list 
    eg. ['a', 'g']
    """

    # Create dictionary to store merge results for each user
    data = {}
    for c in initialsToMerge:
        data[c] = []

    # Read data onto dictionary
    for f in dataFolder.iterdir():
        if f.is_file:
            d = pd.read_csv(f).iloc[:, -1]    # Read only column corresponding to values
            data[f.name[0]].append(d)

    # Pass lists into arrays
    for key in data: data[key] = np.concatenate(data[key])

    # Save onto npz file
    savePath = currentPath / (dataFolder.name+".npz")
    np.savez(savePath, **data)

    # Print shape of saved arrays 
    for k in data: print(f"{k} : {data[k].shape}")


if __name__ == "__main__":
    p = currentPath / "second_collection"
    mergeFiles(p, ['A', 'G', 'J'])
