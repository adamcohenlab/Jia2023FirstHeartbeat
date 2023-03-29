from skimage.color import rgb2gray
from skimage import img_as_ubyte, filters, measure
import skimage.morphology as morph
import scipy.stats as stats
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy.fft import fft, fftfreq
import numpy as np
from .. import utils
from ..analysis import images

def despeckle(img, channels=[0], filter_size=3):
    flatdisk = morph.disk(filter_size)
    flatdisk = flatdisk.reshape([1]*(len(img.shape)-3) + list(flatdisk.shape))
    print(img.shape)
    despeckled_img = np.copy(img)
    if len(channels) > 0:
        for channel in channels:
            print(channel)
            despeckled_img[:,:,channel,:,:] = ndimage.median_filter(img[:,:,channel,:,:], footprint=flatdisk)
    return despeckled_img

def subtract_background(img, channels=[0], median_filter=False, filter_size=3):
    """ 
    Background subtraction according to paper that I don't remember
    """

    bg_subtracted_img = np.copy(img)
    n_timepoints = img.shape[0]

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
            bg_subtracted_img[t,:,c,:,:] = np.maximum(np.zeros(curr_timepoint.shape), img[t,:,c,:,:] - cutoff)
        
        if median_filter:
            flatdisk = morph.disk(filter_size)
            flatdisk = flatdisk.reshape([1,1] + list(flatdisk.shape))
            bg_subtracted_img[:,:,c,:,:] = ndimage.median_filter(bg_subtracted_img[:,:,c,:,:], footprint=flatdisk)
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

def subtract_photobleach(img, n_to_sample=3, channels=[0], filter_size=3):
    means = np.zeros((img.shape[0], img.shape[2]))
    for c in range(img.shape[2]):
        for t in range(img.shape[0]):
            curr_frame = img[t,:,c,:,:]
            means[t,c] = np.mean(curr_frame[curr_frame>0])
    
    slopes = []
    flatdisk = morph.disk(filter_size)
    flatdisk = flatdisk.reshape([1,1] + list(flatdisk.shape))
    
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
                # subtracted_img[t,:,c,:,:] = filters.median(subtracted_img[t,:,c,:,:], selem=flatdisk)
            subtracted_img[:,:,c,:,:] = ndimage.median_filter(subtracted_img[:,:,c,:,:],footprint=flatdisk)

    return subtracted_img


def extract_region_data(data, mask, region):
    global_coords = np.argwhere(mask==region)
    region_data = np.zeros((data.shape[0], global_coords.shape[0]))
    for px_idx in range(global_coords.shape[0]):
        px = global_coords[px_idx]
        region_data[:,px_idx] = data[:,px[0],px[1]]
    return region_data, global_coords

def extract_cropped_region_image(global_coords, intensity):
    global_coords_rezeroed = global_coords - np.min(global_coords, axis=0)
    img = np.zeros(np.max(global_coords_rezeroed, axis=0)+1)
    for idx in range(len(intensity)):
        px = global_coords_rezeroed[idx,:]
        img[px[0], px[1]] = intensity[idx]
    return img


def segment_hearts(img, expected_embryos, prev_coms=None, prev_mask_labels=None, fill_missing=True):
    """ Segment hearts using the following approach:
    1. Identify temporally variable regions of image using coefficient of variation and 
    coarsen using morphological operations. This will roughly identify all calcium activity
    in the FOV.
    2. Perform PCA on each region identified in #1 separately. Take the dot product of the
    first principal component with the intensity-normalized raw data.
    3. Take the Fourier transform of the resulting trace and measure the fraction
    of the total power within the expected frequency band for the heartbeat. If this is above a
    certain threshold (.4 seems to work well), keep the region. Otherwise toss it.
    """
    mean_img = img.mean(axis=0)
    std_img = np.std(img, axis=0)
    cv_img = std_img/mean_img
    cv_mask = cv_img > np.percentile(cv_img,85)
    
    xx = morph.binary_opening(cv_mask, selem= np.ones((5,5)))
    xxx = morph.binary_dilation(xx, selem= np.ones((15,15)))
    labelled = measure.label(xxx)
    rd, gc = images.extract_all_region_data(img, labelled)
    
    bp_counter = 0
    new_mask = np.zeros_like(mean_img, dtype=bool)
    for i, regiondata in enumerate(rd):
        pca = PCA(n_components=5)
        rd_norm = regiondata-np.mean(regiondata, axis=0)
        rd_norm = rd_norm/np.max(np.abs(rd_norm), axis=0)
        pca.fit(rd_norm)
        c = pca.components_[0]
        trace = np.matmul(rd_norm, c)
        N = len(trace)
        yf = fft(trace)
        xf = fftfreq(len(trace), 0.102)[:N//2]
        abs_power = np.abs(yf[0:N//2])
        norm_abs_power = abs_power/np.sum(abs_power)
        band_power = np.sum(norm_abs_power[(xf>0.1) & (xf<1)])
        if band_power > 0.38:
            bp_counter+=1
            comp_abs = np.abs(c)
            correct_indices = comp_abs > filters.threshold_otsu(comp_abs)
            mask_coords = gc[i][correct_indices]
            new_mask[tuple(zip(*mask_coords.tolist()))] = 1
    new_mask = morph.binary_opening(new_mask, selem=np.ones((5,5)))
    new_mask = morph.binary_dilation(new_mask, selem=np.ones((5,5)))
    
    new_mask_labels = measure.label(new_mask)

    coms = ndimage.center_of_mass(new_mask, labels=new_mask_labels, index=np.arange(1,bp_counter+1))
    coms = np.array(coms)
    print(len(coms))
    if len(coms) > expected_embryos:
        plt.imshow(new_mask_labels)
        raise Exception("Extra segments found")
    if prev_coms is not None:
        new_coms_ordered = np.zeros((max(coms.shape[0], prev_coms.shape[0]), 2))
        n_new_rois = 0
        new_mask_copy = np.zeros_like(new_mask_labels, dtype=np.uint8)
        for idx in range(coms.shape[0]):
            com = coms[idx,:]
            dist = np.sum(np.power(com - prev_coms,2), axis=1)
            min_idx = np.argmin(dist)
            if np.sqrt(dist[min_idx]) < 10:
                new_mask_copy[new_mask_labels==(idx+1)] = min_idx+1
                new_coms_ordered[min_idx,:] = com
            elif len(prev_coms) + n_new_rois < expected_embryos:
                n_new_rois += 1
                new_label_val = len(prev_coms) + n_new_rois
                new_mask_copy[new_mask_labels==(idx+1)] = new_label_val
                new_coms_ordered[new_label_val-1,:] = com
        if len(coms) < len(prev_coms) and fill_missing:
            new_labels = set(np.unique(new_mask_copy).tolist())
            old_labels = np.unique(prev_mask_labels).tolist()
            for ol in old_labels:
                if ol not in new_labels:
                    print(ol)
                    new_mask_copy[prev_mask_labels==ol] = ol
                    new_coms_ordered[ol-1,:] = prev_coms[ol-1,:]
        new_mask_labels = new_mask_copy
        coms = new_coms_ordered
    return new_mask_labels, coms