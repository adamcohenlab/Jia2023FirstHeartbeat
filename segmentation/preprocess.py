from skimage.color import rgb2gray
from skimage import img_as_ubyte
import skimage.filters as filters
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import numpy as np
from .. import utils

def subtract_background(img, channels=[0]):
    """ 
    Background subtraction according to paper that  I don't remember
    """

    bg_subtracted_img = np.zeros_like(img)
    n_timepoints = img.shape[0]
    flatdisk = np.zeros((1,5,5))
    flatdisk[0,:,:] = 1
    for c in channels:
        curr_channel = img[:,c,:,:,:]

        n_pixels_zero = np.sum(curr_channel == 0, axis=(1,2,3))
        pct_zeros = n_pixels_zero/curr_channel[0].size*100

        # Subtract cutoffs
        for t in range(n_timepoints):
            curr_timepoint = curr_channel[t,:,:,:]
            cutoff = np.percentile(curr_timepoint, pct_zeros[t] + 5*(100-pct_zeros[t])/100)
            bg_values = curr_timepoint[curr_timepoint <= cutoff]
            bg_values = bg_values[bg_values > 0]
            if len(bg_values) > 0:
                cutoff = np.mean(bg_values)


            ## Gaussian blur? Median filter?
            bg_subtracted_img[t,:,:,:] = filters.median(np.maximum(np.zeros(img.shape[1:]), img[t,:,:,:] - cutoff), selem=flatdisk)
    
    return bg_subtracted_img

def normalize_intensities(img, pct=100):
    """
        Clamp and normalize intensities to a percentile
    """
    normalized_img = np.zeros_like(img)
    for channel in range(img.shape[2]):
        normalized_img[:, :, channel, :, :] = np.minimum(np.ones_like(img[:,:,channel,:,:]), img[:,:, channel, :, :]/np.percentile(img[:,:, channel, :,:], pct))
    return normalized_img