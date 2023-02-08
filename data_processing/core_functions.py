# Script to store all of the main functions for cleaning and loading data

import numpy as np


def load_data(dataPath, triggers=True, releases=False):
    data = np.load(dataPath)

    # Check different users
    users = np.unique(np.array([key.split("_")[0] for key in data], dtype=str))

    # Build X data and corresponding labels
    Xraw = []
    yraw = []

    def appendData(key):
        Xraw.append(data[key])
        yraw.append(np.full(len(data[key]), np.argwhere(users==key.split("_")[0])[0]))

    for key in data:
        _, mode = key.split("_")

        if triggers and (mode=="triggers"): appendData(key)
        if releases and (mode=="releases"): appendData(key)

    Xraw = np.concatenate(Xraw)
    yraw = np.concatenate(yraw)
    return Xraw, yraw


