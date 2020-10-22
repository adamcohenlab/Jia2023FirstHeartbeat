import skimage.filters as filters
import skimage.morphology as morph
from scipy import ndimage
import numpy as np
import skfmm

def get_whole_embryo_mask(img, channel="all", close_size=10, multiplier=0.5):
    mask = np.zeros_like(img[:,:,0,:,:], dtype=bool)
    if channel == "all":
        for c in range(img.shape[2]):
            channel_img = img[:,:,c,:,:]
            mask_channel = np.zeros_like(channel_img)
            # Otsu threshold by timepoint?
            mask_channel = channel_img > filters.threshold_otsu(channel_img)*multiplier
            mask = np.logical_or(mask, mask_channel)
    else:
        channel_img = img[:,:,channel,:,:]
        mask = channel_img > filters.threshold_otsu(channel_img)
    
    pad_width = close_size+1
    mask = np.pad(mask, pad_width)
    close_strel = np.zeros((1,1,2*close_size+1,2*close_size+1))
    close_strel[0,0,:,:] = morph.disk(close_size)
    mask = ndimage.binary_closing(mask, close_strel)
    open_strel = np.zeros((1,1,2*(close_size+1)+1,2*(close_size+1)+1))
    open_strel[0,0,:,:] = morph.disk((close_size+1))
    mask = ndimage.binary_opening(mask, open_strel)
    mask = mask[pad_width:-pad_width,pad_width:-pad_width,pad_width:-pad_width,pad_width:-pad_width]

    return mask

def remove_evl(img, m, channels=[0], first_z_evl=False, last_z_evl=False, z_scale=1, depth=1):

    mask = -m.astype(int)
    img_evl_removed = np.copy(img)

    
    for t in range(img.shape[0]):
        time_mask = mask[t,:,:,:]
        time_mask = np.pad(time_mask, depth, constant_values=-1)
        distances = skfmm.distance(time_mask, dx=[z_scale, 1, 1])
        distances = distances[depth:-depth, depth:-depth, depth:-depth]
        for c in channels:
            timepoint = img[t,:,c,:,:]
            timepoint[distances > - depth] = 0
            img_evl_removed[t,:,c,:,:] = timepoint

    return img_evl_removed.astype(np.uint8)

def label_5d(mask):
    regions_labeled_all_times = np.zeros_like(mask, dtype=np.uint16)
    total_regions_count = 0
    for t in range(mask.shape[0]):
        for c in range(mask.shape[2]):
            regions_labeled, n_regions = ndimage.label(mask[t,:,c,:,:])
            regions_labeled_copy = np.copy(regions_labeled)
            regions_labeled_copy[regions_labeled_copy > 0] += total_regions_count
            regions_labeled_all_times[t,:,c,:,:] = regions_labeled_copy
            total_regions_count += n_regions
    return regions_labeled_all_times, total_regions_count