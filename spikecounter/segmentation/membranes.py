import skimage.filters as filters
import skimage.morphology as morph
from scipy import ndimage
import numpy as np


def raw_membrane_to_mask(stack, erode=True):
    mask = np.zeros_like(stack, dtype=bool)
    objectness = slice_by_slice_objectness(stack)
    for t in range(stack.shape[0]):
        mask[t,:,:,:] = objectness[t,:,:,:] > filters.threshold_triangle(objectness[t,:,:,:])
    if erode:
        strel = morph.disk(1)
        strel = strel.reshape([1,1] + list(strel.shape))
        mask = ndimage.binary_closing(mask, structure=strel)
        mask = ndimage.binary_erosion(mask, structure=strel)
        # for z in range(mask.shape[0]):            
        #     mask[z,:,:] = morph.binary_closing(mask[t,z,:,:], strel)
        #     mask[z,:,:] = morph.binary_erosion(mask[t,z,:,:], strel)
    return mask

def slice_by_slice_objectness(stack):
    print(stack.shape)
    objectness = np.zeros_like(stack, dtype=np.float32)
    for t in range(stack.shape[0]):
        for z in range(stack.shape[1]):
            curr_slice = stack[t,z,:,:]
            frang = filters.frangi(curr_slice, black_ridges=False, sigmas=np.arange(0.5,3,0.5), alpha=1, beta=1,gamma=1)
            objectness[t,z,:,:] = frang
    return objectness