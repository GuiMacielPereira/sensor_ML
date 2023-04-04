
# Notebook to explore transforms of signals
#%%

from core_functions import SensorSignals
import numpy as np
import matplotlib.pyplot as plt

dataPath = "./second_collection_triggs_rels_32.npz"
S = SensorSignals(dataPath) 

idxs = np.random.randint(0, len(S.Xraw), size=5)
X = S.Xraw[idxs]

def plot_samples(X):
    plt.figure(figsize=(12,2))
    for i, x in enumerate(X):
        plt.subplot(1, 5, i+1)
        plt.plot(np.arange(len(x[0])), x[0], "b.")

plot_samples(X)

#%%
# Look at difference between points
Xdiff = np.diff(X, axis=-1)
plot_samples(Xdiff)
Xdiff = np.pad(Xdiff, pad_width=((0, 0), (0, 0), (0, 1)))
print(X.shape, Xdiff.shape)

#%%
Xdiff2 = np.diff(X, n=2, axis=-1)
plot_samples(Xdiff2)
print(X.shape, Xdiff2.shape)

#%%
# Taking the difference every two points is similar to using diff
Xsub2 = X[:, 0, 2:] - X[:, 0, :-2]
Xsub2 = Xsub2[:, np.newaxis, :]
print(Xsub2.shape)
plot_samples(Xsub2)

#%%
# Gradient 
# Seems to smooth out the signal, not a good one to use
Xgrad = np.gradient(X, axis=-1)
plot_samples(Xgrad)
print(X.shape, Xgrad.shape)

#%%
# Fourier
# Not much information included, should probably avoid it
Xfft = np.fft.ifft(X, axis=-1)
plot_samples(Xfft)
