
#%%
# N]otebook for collecting long time intervals from raw data
import numpy as np
import matplotlib.pyplot as plt

fileName = "second_collection"
data = np.load((fileName+".npz"))

f_data = {}
for key in data:
    sig = data[key]
    print("User: ", key)
    print("Total number of points: ", len(sig))
    idxs = np.arange(0, len(sig), step=100)   # Discard the last index so lengths match

    X = []
    for i in idxs:
        cut = sig[i : i+1024]
        if len(cut)==1024:
            if np.any(cut>=2):
                # if np.sum(np.diff(np.argwhere(sig<=0.05)[:, 0])>1) >= 2:
                X.append(cut)

    f_data[key] = np.vstack(X) 
    print("Shape of data: ", f_data[key].shape)

#%%
# Print some random samples
key = "A"
n_samp = 10
plt.figure()
r_idxs = np.random.randint(0, len(f_data[key]), size=n_samp)
for i, idx in enumerate(r_idxs):
    s = f_data[key][idx]
    plt.subplot(n_samp, 1, i+1)
    plt.plot(np.arange(s.size), s, "b.")
plt.show()

#%%
# Save data to file 
output = fileName + "_long_data" + f"_{s.size}.npz"
np.savez(output, **f_data )
print("Saved data!")


