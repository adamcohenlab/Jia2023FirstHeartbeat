import numpy as np
import scipy.ndimage as ndi
from scipy import signal, stats
import matplotlib.pyplot as plt
from . import traces
from sklearn.decomposition import PCA

def refine_segmentation_pca(img, mask, n_components=10, threshold_percentile=70):
    """ Refine manual segmentation to localize the heart using PCA.
    Returns cropped region images and masks for each unique region in the global image mask.
    
    """
    def pca_component_select(pca):
        """ TBD: a smart way to decide the number of principal components. for now just return 1.
        """
        n_components = 1
        selected_components = np.zeros_like(pca.components_[0])
        fraction_explained_variance = pca.explained_variance_/np.sum(pca.explained_variance_)
        for comp_index in range(n_components):
            selected_components = selected_components + pca.components_[comp_index]*fraction_explained_variance[comp_index]        
        return selected_components
    
    region_masks = []
    
    region_data, region_pixels = get_all_region_data(img, mask)
    for region_idx in range(len(region_data)):
        rd = region_data[region_idx]
        rd_bgsub = rd - np.mean(rd, axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(rd_bgsub)
        selected_components = np.abs(pca_component_select(pca))
        
        mask = generate_cropped_region_image(selected_components > np.percentile(selected_components, threshold_percentile))
        selem = np.ones((3,3), dtype=bool)
        mask = morph.binary_opening(mask, selem)
        region_masks.append(mask)
        
        
    return region_masks

def get_all_region_data(img, mask):
    """ Turn all mask regions into pixel-time traces of intensity
    """
    region_pixels = []
    region_pixels = []
    regions = np.unique(mask)
    regions = regions[regions >= 1]
    
    for region in regions:
        rd, gc = images.get_region_data(img, mask, region)
        region_pixels.append(gc)
        region_data.append(rd)
    return region_data, region_pixels
    
def get_region_data(img, mask, region_index):
    """ Turn raw image data from a specific region defined by integer-valued mask into a 2D matrix 
    (timepoints x pixels)
    
    """
    global_coords = np.argwhere(mask==region_index)
    region_data = np.zeros((img.shape[0], global_coords.shape[0]))
    for px_idx in range(global_coords.shape[0]):
        px = global_coords[px_idx]
        region_data[:,px_idx] = img[:,px[0],px[1]]
    return region_data, global_coords

def generate_cropped_region_image(intensity, global_coords):
    """ Turn an unraveled list of intensities back into an image based on the bounding box of
    the specified global coordinates
    
    """
    global_coords_rezeroed = global_coords - np.min(global_coords, axis=0)
    if len(intensity.shape) == 1:
        img = np.zeros(np.max(global_coords_rezeroed, axis=0)+1)
        for idx in range(len(intensity)):
            px = global_coords_rezeroed[idx,:]
            img[px[0], px[1]] = intensity[idx]
    elif len(intensity.shape) == 2:
        img = np.zeros([intensity.shape[0]] + list(np.max(global_coords_rezeroed, axis=0)+1))
        for idx in range(intensity.shape[1]):
            px = global_coords_rezeroed[idx,:]
            img[:,px[0], px[1]] = intensity[:, idx]
    return img

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

def spike_triggered_average_video(img, spike_ends, offset=1, first_spike_index=0, last_spike_index=0, fs=1, sta_length=300):
    """ Create a spike-triggered average video
    """
    spike_triggered_images = []
    for edge in spike_ends[first_spike_index:last_spike_index]:
        if edge+offset+sta_length > img.shape[0]:
            break
        spike_triggered_images.append(img[edge+offset:edge+offset+sta_length,:,:])
    spike_triggered_images = np.array(spike_triggered_images)
    sta = np.mean(spike_triggered_images, axis=0)
    return sta

def test_isochron_detection(vid, x, y, savgol_window=25, figsize=(8,3)):
    """ Check detection of half-maximum in spike_triggered averages
    """
    
    trace = vid[:,y,x]
    trace_smoothed = signal.savgol_filter(trace, savgol_window, 2)
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(trace)
    ax1.plot(trace_smoothed)
    zeroed = trace_smoothed - trace_smoothed[0]
    chron = np.argwhere(zeroed > np.max(zeroed)/2).ravel()[0]
    plt.plot(chron, trace_smoothed[chron], "rx")
    return fig1, ax1

def generate_isochron_map(vid, savgol_window=25, dt=1):
    """ Generate image marking isochrons of wave propagation
    """
    chron = np.zeros(vid.shape[1:])
    for i in range(vid.shape[1]):
        for j in range(vid.shape[2]):
            trace = vid[:,i, j]
            trace_smoothed = signal.savgol_filter(trace, savgol_window, 2)
            zeroed = (trace_smoothed - trace_smoothed[0])
            normed = zeroed/np.max(zeroed)
            time_indices = np.argwhere(normed > 0.5).ravel()

            if len(time_indices) == 0:
                chron[i,j] = vid.shape[0]*dt*1000
            else:
                y = time_indices[0]
                rise_time = (y-1 + (0.5 - normed[y-1])/(normed[y] - normed[y-1]))*dt*1000
                chron[i,j] = rise_time
    return chron

def analyze_wave_prop(masked_image, mask, nbins=16):
    """ Measure wave speed, direction of data
    """
    kernel_size=3
    kernel = np.ones((kernel_size,kernel_size))/kernel_size**2
    filtered = ndi.gaussian_filter(masked_image, [0,2,2])
    isochron = generate_isochron_map(filtered, savgol_window=5, dt=1/10.2)
    isochron_smoothed = ndi.convolve(isochron,kernel)
    isochron_smoothed[~mask] = np.nan
    gradient_y, gradient_x = np.gradient(isochron_smoothed)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    angle = np.arctan2(gradient_y, gradient_x)
    mag_masked = magnitude[mask]
    angle_masked = angle[mask]
    remove_nans = (~np.isnan(mag_masked)) & (~np.isnan(angle_masked))
    mag_masked = mag_masked[remove_nans]
    angle_masked = angle_masked[remove_nans]
    cutoff = np.percentile(mag_masked, 95)
    mag_masked[mag_masked > cutoff] = cutoff
    weighted = angle_masked*mag_masked
    direction = np.sum(weighted)/np.sum(mag_masked)
    mean_magnitude = np.mean(mag_masked)
    var_magnitude = np.var(mag_masked)
    median_magnitude = np.median(mag_masked)
    hist, _ = np.histogram(angle_masked, bins=nbins, range=(-np.pi/2, np.pi/2))
    entr = stats.entropy(hist)
    return mean_magnitude, var_magnitude, median_magnitude, direction, entr