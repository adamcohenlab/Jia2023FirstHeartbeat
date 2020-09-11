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

    bg_subtracted_img = np.copy(img)
    n_timepoints = img.shape[0]
    flatdisk = np.zeros((1,5,5))
    flatdisk[0,:,:] = 1
    for c in channels:
        curr_channel = img[:,:,c,:,:]

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
            bg_subtracted_img[t,:,c,:,:] = filters.median(np.maximum(np.zeros(curr_timepoint.shape), img[t,:,c,:,:] - cutoff), selem=flatdisk)
    
    return bg_subtracted_img

def normalize_intensities(img, pct=100, scale=1):
    """
        Clamp and normalize intensities to a percentile
    """
    if type(pct) is (int or float):
        pct = [pct]*img.shape[2]
    
    normalized_img = np.zeros_like(img)
    for channel in range(img.shape[2]):
        normalized_img[:, :, channel, :, :] = np.minimum(np.ones_like(img[:,:,channel,:,:]), img[:,:, channel, :, :]/np.percentile(img[:,:, channel, :,:], pct[channel]))*scale
    return normalized_img

def normalize_intensities_maxproj(img, pct=100, scale=1):
    """
        Clamp and normalize intensities to a percentile
    """
    if type(pct) is (int or float):
        pct = [pct]*img.shape[1]
    
    normalized_img = np.zeros_like(img)
    for channel in range(img.shape[1]):
        normalized_img[:, channel, :, :] = np.minimum(np.ones_like(img[:, channel,:,:]), img[:, channel, :, :]/np.percentile(img[:, channel, :,:], pct[channel]))*scale
    return normalized_img

def subtract_photobleach(img, n_to_sample=3, channels=[0], filter_size=5):
    means = np.zeros((img.shape[0], img.shape[2]))
    for c in range(img.shape[2]):
        for t in range(img.shape[0]):
            curr_frame = img[t,:,c,:,:]
            means[t,c] = np.mean(curr_frame[curr_frame>0])
    
    slopes = []
    flatdisk = np.zeros((1,filter_size,filter_size))
    flatdisk[0,:,:] = 1
    
    ## TODO - fit this scaling locally to account for cell movements
    for c in range(img.shape[2]):
        # Should this be an exponential? Probably
        slope, intercept, r, p, _ = stats.linregress(np.arange(img.shape[0]), y=means[:,c])
        slopes.append(slope)
        print(c, slope, intercept, r, p)

    initial_intensities = np.mean(img[:n_to_sample,:,:,:,:], axis=0)
    scaling = initial_intensities
    subtracted_img = np.zeros_like(img)
    for c in range(img.shape[2]):
        if c in channels:
            print(c)
            scaling[:,c,:,:] = initial_intensities[:,c,:,:]/means[0,c]
            for t in range(img.shape[0]):
                subtracted_img[t,:,c,:,:] = np.maximum(np.zeros_like(img[t,:,c,:,:]), img[t,:,c,:,:] - scaling[:,c,:,:]*t*slopes[c])
                subtracted_img[t,:,c,:,:] = filters.median(subtracted_img[t,:,c,:,:], selem=flatdisk)
        else:
            subtracted_img[:,:,c,:,:] = img[:,:,c,:,:]
    

    return subtracted_img