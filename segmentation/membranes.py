import skimage.filters as filters
import skimage.morphology as morph
import numpy as np


def raw_membrane_to_mask(stack, erode=True):
    mask = np.zeros_like(stack, dtype=bool)
    objectness = slice_by_slice_objectness(stack)
    mask = objectness > filters.threshold_triangle(objectness)
    if erode:
        strel = morph.disk(1)
        for z in range(mask.shape[0]):            
            mask[z,:,:] = morph.binary_closing(mask[z,:,:], strel)
            mask[z,:,:] = morph.binary_erosion(mask[z,:,:], strel)
    return mask

def slice_by_slice_objectness(stack):
    objectness = np.zeros_like(stack, dtype=np.float32)
    for z in range(stack.shape[0]):
        curr_slice = stack[z,:,:]
        medfiltered = filters.median(curr_slice, selem=np.ones((5,5)))
        frang = filters.frangi(medfiltered, black_ridges=False, sigmas=np.arange(0.5,3,0.5), alpha=1, beta=1,gamma=1)
        objectness[z,:,:] = frang
    return objectness