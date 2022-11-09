
#%%
# Exploring preparation of data set

#%%
# Load data
import numpy as np
import matplotlib.pyplot as plt
data = np.load("./sorted_data.npz")


#%%
# Find signal triggers
def findTrigIdxs(signal, threshold=0.1):
    """
    Finds the indexes of the signal where the trigger happens
    """

    mask = signal <= threshold
    mask = mask[:, np.newaxis]
    zerosIdx = np.argwhere(mask)[:, 0].astype(int)
    jumpIdx = np.argwhere(np.diff(zerosIdx)>1)[:, 0].astype(int)
    trigIdx =  zerosIdx[jumpIdx].astype(int)
    return trigIdx

def separateIntoTriggers(signal, trigIdx, width=30):
    triggers = []
    for i in trigIdx:
        triggers.append(signal[i:i+width])
    return np.array(triggers)


def plotTriggers(triggers, name):
    plt.figure(figsize=(15,15))
    plt.suptitle(name)
    N = len(triggers)
    for i in range(N):
        if i+1 > int(np.sqrt(N))**2: continue    # Skip some signals to avoid error due to outside of grid
        plt.subplot(int(np.sqrt(N)), int(np.sqrt(N)), i+1)
        trig = triggers[i]
        plt.plot(range(len(trig)), trig, "b.")

def plotSignal(signal):
    plt.figure()
    plt.plot(range(len(signal)), signal, "b.")
    plt.show()
#%%
# Look at triggers

key = "C1"
signal = data[key][0]
print(signal.shape)
trigIdx = findTrigIdxs(signal)
triggers = separateIntoTriggers(signal, trigIdx)
plotSignal(signal)
plotTriggers(triggers, key) 

#%%
key = "C2"
signal = data[key][0]
trigIdx = findTrigIdxs(signal, threshold=1)
triggers = separateIntoTriggers(signal, trigIdx)
plotSignal(signal)
plotTriggers(triggers, key) 

#%%
# Filter data into triggers and save it
filteredData = {}
for key in data:
    signal = data[key].flatten()
    trigIdx = findTrigIdxs(signal, threshold=1)
    triggers = separateIntoTriggers(signal, trigIdx)
    filteredData[key] = triggers
    print("saving ", triggers.shape)

savePath = "./triggers_data.npz"
np.savez(savePath, **filteredData)


