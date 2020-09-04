#! /usr/bin/python3
import numpy as np
import scipy.ndimage as nd
import skimage.io as skio

# axes are T,(Z), (C), (X), (Y) 

def space_average_over_time(timeseries, mask=None):
    if mask is not None:
        time_mask = np.zeros_like(timeseries)
        print(time_mask.shape)
        time_mask[0,:,:] = mask
        for t in range(1, time_mask.shape[0]):
            mask[mask!=0] += 1
            time_mask[t,:,:] = mask
        print(time_mask.max())
    return np.array(nd.mean(timeseries, labels=time_mask, index=np.arange(1, time_mask.shape[0]+1)))


# n_traces_to_sample = 10
# for i in range(n_traces_to_sample):
#     idx_to_sample = np.random.randint(spike_detected.shape[0])
#     fig1, ax1 = plt.subplots(figsize=(10,6))
#     coords = spike_detected[idx_to_sample,:]
#     trace = img[:,coords[1], coords[2], coords[3]]
#     ax1.plot(trace)
#     ax1.set_title(str(coords))
# plt.show()