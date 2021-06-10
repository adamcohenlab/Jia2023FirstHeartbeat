import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from . import traces

def spike_mask_to_stim_index(spike_mask):
    """ Convert detected spikes from crosstalk into a end of stimulation index for spike-triggered averaging

    """
    diff_mask = spike_mask[1:].astype(int) - spike_mask[:-1].astype(int)
    stim_end = np.argwhere(diff_mask==-1).ravel()+1
    return stim_end


def image_to_trace(img, mask = None):
    """ Convert part of an image to a trace according to a mask
    
    """
    if mask is None:
        trace = img.mean(axis=(1,2))
    else:
        masked_img = np.ma.masked_array(img, mask=~mask)
        trace = masked_img.mean(axis=(1,2))
    return trace

def plot_image_mean_and_stim(img, mask=None, style="line", duration=0, fs=1):
    """ Plot mean of image (mask optional) and mark points where the image was stimulated

    """
    trace = image_to_trace(img, mask)
    masked_trace, spike_mask = traces.remove_stim_crosstalk(trace)
    stim_end = spike_mask_to_stim_index(spike_mask)
    fig1, ax1 = plt.subplots(figsize=(12,6))
    ts = np.arange(img.shape[0])/fs
    ax1.plot(ts, trace, color="C1")
    bot, top = ax1.get_ylim()
    if style == "line":
        ax1.vlines(stim_end/fs, ymin=bot, ymax=top)
    elif style == "rect":
        raise ValueError("Rectangle stimulation to be implemented")
    else:
        raise ValueError("Style should be line or rect")
    return fig1, ax1


def background_subtract(img, dark_level=100):
    return img - dark_level


def get_image_dFF(img, baseline_percentile=10):
    """ Convert a raw image into dF/F

    """
    baseline = np.percentile(img, baseline_percentile, axis=0)
    dFF = img/baseline
    return dFF

def filter_and_downsample(img, filter_function, sampling_factor=2):
    """ filter in time and downsample

    """
    filtered_img = np.apply_along_axis(filter_function, 0, img)
    filtered_img = filtered_img[np.arange(filtered_img.shape[0], step=sampling_factor),:,:]
    return filtered_img

def spike_triggered_average_video(img, offset=1, first_spike_index=0, last_spike_index=0, fs=1, sta_length=300):
    """ Create a spike-triggered average video

    Returns:
    sta - a spike triggered average video of frames defined by sta_length
    stim_image - the region of stimulus as observed by frame captured during DMD switching
    """
    mean_img = image_to_trace(img, mask=None)
    _, spike_mask = traces.remove_stim_crosstalk(mean_img)
    stim_end = spike_mask_to_stim_index(spike_mask)

    spike_triggered_images = []
    for edge in stim_end[first_spike_index:last_spike_index]:
        if edge+offset+sta_length > img.shape[0]:
            break
        spike_triggered_images.append(img[edge+offset:edge+offset+sta_length,:,:])
    spike_triggered_images = np.array(spike_triggered_images)
    sta = np.array(spike_triggered_images)
    stim_image = np.mean(img[stim_end,:,:], axis=0)
    return sta, stim_image