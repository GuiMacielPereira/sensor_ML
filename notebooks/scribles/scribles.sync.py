#%%
# --------- Cells with some scribles ------------

# This function can be used to look at a sliding window 
# Not better performing than just using a split (i.e. no stride==input_size)
def change_input(x, I, S):
    """I is window size, S is stride"""

    # Make a mask used to transform inputs 
    bs = x.shape[0]
    W = x.shape[-1] 
    L = int((W - I) / S + 1)

    mask = np.full((L, W), False)
    mask[0, :I] = True 
    for i in range(1, L):
        mask[i] = np.roll(mask[i-1], shift=S)

    result = np.zeros((bs, L, I))
    x = x * np.ones((1, L, 1))  # multiply by ones to extend shape
    for i in range(bs):
        result[i] = x[i][mask].reshape((L, I))
    return result

#%%
# Exploring some reshaping
import numpy as np
x = np.arange(50).reshape((10, 1, 5))
print(x)
res = []
input_size = 5
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x)
# for i in range(x.shape[-1] - input_size + 1):
#     res.append(x[:, :, i:i+input_size])
# x = np.concatenate(res, axis=1)
# print("Reshaped:\n", x)
# print(x.shape)

#%%
# Looking into grouping triggers of groups of 3
import numpy as np
n = 3
x = np.arange(50).reshape((10, 1, 5))

bs, _, s = x.shape
assert n<=bs, "Cannot form groups bigger than the number of triggers available"
x = x[: bs-(bs%n), :, :]
print(x.shape)

x = x.reshape(-1, n, s)
print(x.shape)

print(x)

#%%
import numpy as np
x = np.arange(50).reshape((10, 1, 5))
n = 3  
res = []
for i in range(len(x) - n):
    res.append(x[i:i+n, 0, :])
print(x)
x = np.array(res)
print(x)
print(x.shape)

#%%
import sklearn
x = np.arange(50).reshape((5, 2, 5))
print(x)
batch_size, _, input_size = x.shape
x = x.reshape(-1, input_size)
print(x)
x = sklearn.utils.shuffle(x)
x = x.reshape(batch_size, -1, input_size)
print(x)


#%%
import numpy as np
x = np.arange(50).reshape((5, 2, 5))

print(x)
shape = x.shape
x = x.reshape(shape[0], -1)
x = x.reshape(shape)
print(x)




