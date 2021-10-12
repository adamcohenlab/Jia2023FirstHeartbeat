import numpy as np
import scipy.ndimage as ndi
import skimage.io as skio
from scipy import signal, stats, interpolate, optimize
import matplotlib.pyplot as plt
from skimage import transform, filters, morphology

from . import traces
from .. import utils
from ..ui import visualize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def refine_segmentation_pca(img, mask, n_components=10, threshold_percentile=70):
    """ Refine manual segmentation to localize the heart using PCA assuming that transients are the largest fluctuations present.
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
    region_data = []
    
    for region in regions:
        rd, gc = get_region_data(img, mask, region)
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

def display_pca_data(pca, raw_data, gc, n_components=5):
    """ Show spatial principal components of a video and the corresponding temporal trace (dot product).
    """
    for i in range(n_components):
        fig1, axes = plt.subplots(1, 2, figsize=(12,6))
        axes = axes.ravel()
        comp = pca.components_[i]
        cropped_region_image = generate_cropped_region_image(comp, gc)
        tr = np.matmul(raw_data, comp)
        axes[0].imshow(cropped_region_image)
        axes[0].set_title("PC %d (Fraction Var: %.3f)" %(i+1, pca.explained_variance_ratio_[i]))
        axes[1].set_title("PC Value")
        axes[1].plot(tr)

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

def interpolate_invalid_values(img, mask):
    """ Interpolate invalid values pixelwise
    """
    xs = np.arange(img.shape[0])[~mask]
    invalid_filled = np.copy(img)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            ys = img[:,i,j][~mask]
            interp_f = interpolate.interp1d(xs, ys, kind="cubic", fill_value="extrapolate")
            missing_xs = np.argwhere(mask).ravel()
            missing_ys = interp_f(missing_xs)
            invalid_filled[mask,i,j] = missing_ys
    return invalid_filled
    

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

def get_spike_kernel(img, kernel_length, nbefore, peak_prominence, savgol_length=51):
    """ Estimate a temporal spike kernel by doing naive peak detection and selecting a temporal window. Pick a subsample with the largest peak amplitudes after smoothing (presumably largest SNR) and average.
    
    Returns:
        kernel - temporal kernel for the spike
        kernel_hits - which pixels in the image were selected to be averaged into the kernel
    """
    kernel_candidates = []
    height = img.shape[1]
    width = img.shape[2]
    kernel_hits = np.zeros((height, width), dtype=bool)

    for i in range(height):
        for j in range(width):
            trace_smoothed = signal.savgol_filter(img[:,i,j], savgol_length, 2)
            px_peaks, _ = signal.find_peaks(trace_smoothed, prominence=peak_prominence)
            if len(px_peaks) > 1:
                plt.plot(trace_smoothed)
                plt.plot(px_peaks, trace_smoothed[px_peaks], "rx")
                raise ValueError("More than one candidate peak detected")
            elif len(px_peaks) == 1:
                kernel_hits[i,j] = True
                pidx = px_peaks[0]
                if pidx < nbefore:
                    xs = np.arange(nbefore-pidx, kernel_length)
                    ys = img[:pidx-nbefore+kernel_length, i,j]
                    interpf = interpolate.interp1d(xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                elif pidx-nbefore+kernel_length > img.shape[0]:
                    xs = np.arange(img.shape[0] - (pidx-nbefore))
                    ys = img[pidx-nbefore:,i,j]
                    interpf = interpolate.interp1d(xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                else:
                    ktrace = img[pidx-nbefore:pidx-nbefore+kernel_length,i,j]
                ktrace = (ktrace - np.min(ktrace))/(np.max(trace_smoothed) - np.min(trace_smoothed))
                kernel_candidates.append(ktrace)
    kernel = np.zeros(img.shape[0])
    kernel[:kernel_length] = np.mean(np.array(kernel_candidates[:]), axis=0)
    kernel[kernel_length:] = kernel[kernel_length-1]
    return kernel, kernel_hits

def snapt(img, kernel, offset_width=0):
    """ Run SNAPT fitting algorithm (Hochbaum et al. Nature Methods 2014)
    """
    height = img.shape[1]
    width = img.shape[2]
    beta = np.zeros((height, width, 4))
    error_det = np.zeros((height, width))
    failed_counter = 0
    t0 = np.argmax(kernel)
    L = len(kernel)
    minshift = - t0
    maxshift = (L - t0)
    
    beta = np.apply_along_axis(lambda tr: kernel_fit_single_trace(tr, kernel, minshift, maxshift, offset_width), \
                               0, img)
    error_det = beta[4,:,:]
    beta = beta[:4,:,:]
    failed_pixels = np.sum(np.isnan(error_det))
    print("%d/%d pixels failed to fit (%.2f %%)" % (failed_counter, height*width, failed_counter/(height*width)*100))
    
    
#     for i in range(height):
#         for j in range(width):
#             try:
#                 popt, pcov = optimize.curve_fit(utils.shiftkern, kernel, img[:,i,j], p0=[1,1,1,np.random.randint(-offset_width,offset_width+1)], absolute_sigma=True, \
#                                                 bounds=([0,-np.inf,0,minshift],[np.inf,np.inf,np.inf,maxshift]))
#                 beta[i,j,:] = popt
#                 error_det[i,j] = np.linalg.det(pcov)
#             except Exception as e:
#                 print("(%d, %d) %s" % (i,j, str(e)))
#                 beta[i,j,:] = np.nan*np.ones(4)
#                 error_det[i,j] = np.nan
#                 failed_counter += 1
#     print("%d/%d pixels failed to fit (%.2f %%)" % (failed_counter, height*width, failed_counter/(height*width)*100))
    return beta, error_det

def kernel_fit_single_trace(trace, kernel, minshift, maxshift, offset_width):
    """ Nonlinear fit of empirical kernel to a single timeseries
    """
    try:
        popt, pcov = optimize.curve_fit(utils.shiftkern, kernel, trace,\
                                        p0=[1,1,1,np.random.randint(-offset_width,offset_width+1)], absolute_sigma=True, \
                                        bounds=([0,-np.inf,0,minshift],[np.inf,np.inf,np.inf,maxshift]))
        beta = popt
        error_det = np.linalg.det(pcov)
    except Exception as e:
        beta = np.nan*np.ones(4)
        error_det = np.nan
    beta = np.append(beta, error_det)
    return beta

def spline_timing(img, s=0.1, n_knots=4):
    knots = np.linspace(0, img.shape[0]-1, num=n_knots)[1:-1]
    beta = np.apply_along_axis(lambda tr: spline_fit_single_trace(tr, s, knots), 0, img)
    return beta

def spline_fit_single_trace(trace, s, knots, plot=False):
    """ Least squares spline fitting of a single timeseries
    """
    x = np.arange(len(trace))
    (t,c,k), res,_,_ = interpolate.splrep(x,trace, s=s, task=-1,t=knots, full_output=True,k=3)
    spl = interpolate.BSpline(t,c,k)
    spl_values = spl(x)
    
    if plot:
        fig1, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(trace)
        ax1.plot(spl(x))
    
    dspl = spl.derivative()
    d2spl = spl.derivative(nu=2)
#     naive_max = np.argmax(spl_values)
#     naive_min = np.argmin(spl_values[:naive_max])
#     e1 = optimize.fsolve(dspl, naive_max-5)
#     e2 = optimize.fsolve(dspl, naive_min+5)
    extrema = np.argwhere(np.diff(np.sign(dspl(x)))).ravel()
#     extrema = np.array([naive_min, naive_max])
            
    extrema_values = spl(extrema)
    if plot:
        ax1.plot(extrema, extrema_values, "ko")
    if np.all(np.logical_or(extrema < 0, extrema > len(trace))):
        beta = np.nan*np.ones(4)
        return beta
    global_max_index = np.argmax(extrema_values)
    global_max = extrema[global_max_index]
    global_max_val = extrema_values[global_max_index]
    
    d2 = d2spl(extrema)
    minima_indices = np.argwhere(d2 > 0).ravel()
    if len(minima_indices) == 0:
        last_min_before_max = 0
    else:
        minima = extrema[minima_indices]
        premax_minima_indices = np.argwhere(minima < global_max).ravel()
        if len(premax_minima_indices) == 0:
            last_min_before_max = 0
        else:
            last_min_before_max = minima[premax_minima_indices[-1]]
    if plot:
        ax1.plot([last_min_before_max, global_max], spl([last_min_before_max, global_max]), "rx")
    min_val = spl(last_min_before_max)
    halfmax_magnitude = (global_max_val - min_val)/2 + min_val
    try:
#         halfmax = optimize.minimize_scalar(lambda x: (spl(x) - halfmax_magnitude)**2, method='bounded', \
#                                        bounds=[last_min_before_max, global_max]).x
        try:
            halfmax = np.argmin((spl_values[last_min_before_max:global_max]-\
                               halfmax_magnitude)**2) + last_min_before_max
        except Exception as e:
            beta = np.nan*np.ones(4)
            return beta            
        if plot:
            ax1.plot(halfmax, spl(halfmax), "gx")
    except Exception as e:
        print(extrema)
        print(d2)
        print(last_min_before_max)
        print(global_max)
        raise e
    beta = np.array([global_max, halfmax, global_max_val/min_val,res])
    
    return beta
    
def correct_photobleach(img, mask=None, method="localmin", nsamps=51):
    """ Perform photobleach correction on each pixel in an image
    """
    if method == "linear":
        corrected_img = np.zeros_like(img)
        for y in range(img.shape[1]):
            for x in range(img.shape[2]):
                corrected_trace, _ = traces.correct_photobleach(img[:,y,x])
                corrected_img[:, y,x] = corrected_trace
                
    elif method == "localmin":
        mean_trace = image_to_trace(img, mask)
        _, pbleach = traces.correct_photobleach(mean_trace, method=method, nsamps=nsamps)
        corrected_img = np.divide(img, pbleach[:,np.newaxis,np.newaxis])
        print(corrected_img.shape)
    else:
        raise ValueError("Not Implemented")
        
    return corrected_img

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

def generate_isochron_map(vid, savgol_window=25, dt=1, prefilter=True):
    """ Generate image marking isochrons of wave propagation
    """
    chron = np.zeros(vid.shape[1:])
    if prefilter:
        smoothed_vid = np.apply_along_axis(lambda x: signal.savgol_filter(x, savgol_window, 2), 0, vid)
    else:
        smoothed_vid = vid
        
    zeroed = smoothed_vid - smoothed_vid[0,:,:]
    normed = zeroed/zeroed.max(axis=0)
    hm_indices = np.apply_along_axis(find_half_max, 0, normed)
    chron = hm_indices*dt*1000
    return chron

def analyze_wave_prop(masked_image, mask, nbins=16, savgol_window=5, dt=1):
    """ Measure wave speed, direction of data
    """
    kernel_size=3
    kernel = np.ones((kernel_size,kernel_size))/kernel_size**2
    filtered = ndi.gaussian_filter(masked_image, [0,2,2])
    
    isochron = generate_isochron_map(filtered, savgol_window=5, dt=dt)
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

def fill_nans_with_neighbors(img):
    isnan = np.isnan(img)

def downsample_video(raw_img, downsample_factor, aa=True):
    """ Downsample video in space with optional anti-aliasing
    """
    if downsample_factor == 1:
        di = raw_img
    else:
        if aa:
            sos = signal.butter(4, 1/downsample_factor, output='sos')
            filtered = np.apply_over_axes(lambda a, axis: np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), axis, a), raw_img, [1,2])
            di = transform.downscale_local_mean(filtered, (1,downsample_factor,downsample_factor))
        else:
            di = transform.downscale_local_mean(raw_img, (1,downsample_factor,downsample_factor))
    return di

def get_image_dff_corrected(img, nsamps_corr, mask=None, plot=None, full_output=False):
    """ Convert image to Delta F/F after doing photobleach correction and sampling 
    """
    
    # Generate mask if not provided
    mean_img = img.mean(axis=0)
    if mask is None:
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0]/50)
        mask = morphology.binary_closing(mask, selem=np.ones((kernel_size, kernel_size)))
    
    if plot is not None:
        _,_ = visualize.display_roi_overlay(mean_img, mask.astype(int), ax=plot)
    # Correct for photobleaching and convert to DF/F
    pb_corrected_img = correct_photobleach(img, mask=np.tile(mask, (img.shape[0], 1, 1)), nsamps=nsamps_corr)
    dFF_img = get_image_dFF(pb_corrected_img)
    
    if full_output:
        return dFF_img, mask
    else:
        return dFF_img
    
    
def image_to_sta(raw_img, downsample_factor=1, fs=1, mask=None, plot=False, savedir=None, prom_pct=90, sta_bounds="auto", aa=True, exclude_pks = None, offset=0):
    """ Convert raw image into spike triggered average
    """
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    if plot:
        fig1, axes = plt.subplots(2,3, figsize=(15,10))
        axes = axes.ravel()

    # Downsample
    di = downsample_video(raw_img, downsample_factor, aa=aa)
    print("Downsample complete")
    dFF_img, mask = get_image_dff_corrected(di, 111, plot=axes[0], full_output=True)
    raw_trace = image_to_trace(di, np.tile(mask, (di.shape[0], 1, 1)))
    
    ss = StandardScaler()
    rd, gc = get_region_data(dFF_img, np.ones((dFF_img.shape[1], dFF_img.shape[2])), 1)
    rd = ss.fit_transform(rd)
    pca_full = PCA(n_components=5)
    pca_full.fit(rd)
    pca_img = generate_cropped_region_image(pca_full.components_[0], gc)
    

    dFF_mean = image_to_trace(dFF_img, mask=np.tile(mask, (di.shape[0], 1, 1)))

    if savedir:
        skio.imsave(os.path.join(savedir, "dFF.tif"), dFF_img)
        skio.imsave(os.path.join(savedir, "pca.tif"), pca_img)
        np.savez(os.path.join(savedir, "dFF_mean.npz"), dFF_mean=dFF_mean)
    
    # Identify spikes
    ps = np.percentile(dFF_mean,[10,prom_pct])
    prominence = ps[1] - ps[0]
    pks, _ = signal.find_peaks(dFF_mean, prominence=prominence)
    
    if exclude_pks:
        keep = np.ones(len(pks))
        keep[exclude_pks] = 0
        pks = pks[keep]

    if plot:
        axes[1].plot(np.arange(dFF_img.shape[0])/fs, dFF_mean)
        axes[1].plot(pks/fs, dFF_mean[pks], "rx")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel(r"$F/F_0$")
        tx = axes[1].twinx()
        tx.plot(np.arange(dFF_img.shape[0])/fs, raw_trace, color="C1")
        tx.set_ylabel("Mean counts")
        axes[2].imshow(pca_img)

    # Automatically determine bounds for spike-triggered average
    if sta_bounds == "auto":
        # Align all detected peaks of the full trace and take the mean
        try:
            aligned_traces = traces.align_fixed_offset(np.tile(dFF_mean, (len(pks), 1)), pks)
        except Exception as e:
            return raw_trace
        mean_trace = np.nanmean(aligned_traces, axis=0)
        # Smooth
        spl = interpolate.UnivariateSpline(np.arange(len(mean_trace)), np.nan_to_num(mean_trace, nan=np.nanmin(mean_trace)), s=0.001)
        smoothed = spl(np.arange(len(mean_trace)))
        smoothed_dfdt = spl.derivative()(np.arange(len(mean_trace)))   
        # Find the first minimum before and the first minimum after the real spike-triggered peak
        if plot:
            axes[4].plot(mean_trace)
            axes[4].plot(smoothed, color="C1")
            tx = axes[4].twinx()
            tx.plot(smoothed_dfdt, color="C2")
            tx.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        try:
            minima_left = np.argwhere((smoothed_dfdt < 0)).ravel()
            minima_right = np.argwhere((smoothed_dfdt > 0)).ravel()
            # calculate number of samples to take before and after the peak
            b1 = pks[-1] - minima_left[minima_left<pks[-1]][-1]+offset
            b2 = minima_right[minima_right > pks[-1]][0] - pks[-1]+offset
        except Exception:
            return mean_trace
        
        if plot:
            axes[3].plot([pks[-1]-b1, pks[-1], pks[-1]+b2], mean_trace[[pks[-1]-b1, pks[-1], pks[-1]+b2]], "rx")
        


    # Collect spike traces according to bounds
    spike_traces = np.nan*np.ones((len(pks), b1+b2))
    spike_images = np.nan*np.ones((len(pks), b1+b2, dFF_img.shape[1], dFF_img.shape[2]))
    print(b1,b2)
    for idx, pk in enumerate(pks):
        n_prepend = max(0, b1-pk)
        n_append = max(0, pk+b2-len(dFF_mean))

        mean_block = np.concatenate([np.ones(n_prepend)*np.nan, dFF_mean[max(0, pk-b1):min(len(dFF_mean), pk+b2)], np.ones(n_append)*np.nan])
        img_block = np.concatenate([np.ones((n_prepend, dFF_img.shape[1], dFF_img.shape[2]))*np.nan, \
                                    dFF_img[max(0, pk-b1):min(len(dFF_mean), pk+b2),:,:], np.ones((n_append, dFF_img.shape[1], dFF_img.shape[2]))*np.nan])
        
        spike_traces[idx,:] = mean_block
        spike_images[idx,:,:] = img_block
    
    sta_trace = np.nanmean(spike_traces, axis=0)
    sta = np.nanmean(spike_images, axis=0)

    if plot:
        axes[3].plot(np.arange(b1+b2)/fs, sta_trace)
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel(r"$F/F_0$")
        axes[3].set_title("STA taps: %d + %d" % (b1,b2))
        plt.tight_layout()
    if savedir:
        skio.imsave(os.path.join(savedir, "sta.tif"), sta)
        if plot:
            plt.savefig(os.path.join(savedir, "QA_plots.tif"))
    
    return sta
    
    