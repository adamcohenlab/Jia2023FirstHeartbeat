import numpy as np
import scipy.ndimage as ndi
import skimage.io as skio
from scipy import signal, stats, interpolate, optimize, ndimage
import matplotlib.pyplot as plt
from skimage import transform, filters, morphology, measure, draw
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
import os
import mat73
import warnings
import colorcet as cc
import seaborn as sns
import pickle
from cycler import cycler


from . import traces
from .. import utils
from ..ui import visualize

def load_dmd_target(rootdir, expt_name, downsample_factor=1):
    expt_data = mat73.loadmat(os.path.join(rootdir, expt_name, "output_data_py.mat"))["dd_compat_py"]
    width = int(expt_data["camera"]["roi"][1])
    height = int(expt_data["camera"]["roi"][3])
    
    dmd_target = expt_data["dmd_lightcrafter"]['target_image_space']
    offset_x = int((dmd_target.shape[1] - width)/2)
    offset_y = int((dmd_target.shape[0] - height)/2)
    dmd_target = dmd_target[offset_y:-offset_y,offset_x:-offset_x]
    dmd_target = dmd_target[::downsample_factor,::downsample_factor].astype(bool)
    return dmd_target


def load_image(rootdir, expt_name, subfolder="", raw=True):
    d = os.path.join(rootdir, subfolder)
    
    all_files = os.listdir(d)
    expt_files = 0
    try:
        expt_data = mat73.loadmat(os.path.join(rootdir, expt_name, "output_data_py.mat"))["dd_compat_py"]
    except Exception:
        expt_data = None

    if raw:
        width = int(expt_data["camera"]["roi"][1])
        height = int(expt_data["camera"]["roi"][3])
        try:
            img = np.fromfile(os.path.join(rootdir, subfolder, expt_name, "frames.bin"), dtype=np.dtype("<u2")).reshape((-1,width,height))
        except Exception:
            raw = False
        
    if not raw:
        for f in all_files:
            if expt_name in f and ".tif" in f:
                expt_files +=1
        if expt_files == 1:
            img = skio.imread(os.path.join(d, "%s.tif" % expt_name))
        else:
            img = [skio.imread(os.path.join(d, "%s_block%d.tif" % (expt_name, block+1))) for block in range(expt_files)]
            img = np.concatenate(img, axis=0)
    if expt_data is not None:
        fc_max = np.max(expt_data["frame_counter"])
        if fc_max > img.shape[0]:
            warnings.warn("%d frames dropped" % (fc_max - img.shape[0]))
            last_tidx = img.shape[0]
        else:
            last_tidx = int(min(fc_max, expt_data["camera"]["frames_requested"] - expt_data["camera"]["dropped_frames"]))
    else:
        last_tidx = img.shape[0]
    
    return img[:last_tidx,:,:], expt_data

def load_confocal_image(path, direction="both", extra_offset=2):
    matdata = mat73.loadmat(path)['dd_compat_py']
    confocal = matdata['confocal']
    points_per_line = int(confocal["points_per_line"])
    numlines = int(confocal["numlines"])
    
    img = confocal['PMT'].T.reshape((confocal["PMT"].shape[1], numlines, 2, points_per_line))
    
    if direction == "fwd":
        return img[:,:,0,:]
    
    
    elif direction == "both":
        fwd_scan = img[:,:,0,:]
        rev_scan = img[:,:,1,:]
        
        reshaped_xdata = confocal['xdata'].reshape(numlines, 2,\
                                           points_per_line)
        pixel_step = reshaped_xdata[0,0,1] - reshaped_xdata[0,0,0]
        offset = np.min(np.mean(confocal["galvofbx"][:int(confocal["points_per_line"]),:],axis=1))
        
        offset_pix = offset/pixel_step
        
        offset_idx = int(np.round(-(offset_pix+extra_offset)))
        revscan_shift = np.roll(np.flip(rev_scan, axis=2), offset_idx, axis=2)
        revscan_shift[:,:,:offset_idx] = np.mean(revscan_shift, axis=(1,2))[:,np.newaxis,np.newaxis]
        
        mean_img = (fwd_scan + revscan_shift)/2
        return mean_img
    
    else:
        raise Exception("Not implemented")
def generate_invalid_frame_indices(stim_trace):
    invalid_indices_daq = np.argwhere(stim_trace > 0).ravel()
    invalid_indices_camera = np.concatenate((invalid_indices_daq, invalid_indices_daq+1))
    return invalid_indices_camera

def refine_segmentation_pca(img, rois, n_components=10, threshold_percentile=70):
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
    region_data, region_pixels = get_all_region_data(img, rois)
    for region_idx in range(len(region_data)):
        rd = region_data[region_idx]
        gc = region_pixels[region_idx]
        rd_bgsub = rd - np.mean(rd, axis=0)
        try:
            pca = PCA(n_components=n_components)
            pca.fit(rd_bgsub)
            selected_components = np.abs(pca_component_select(pca))
        
            indices_to_keep = selected_components > np.percentile(selected_components, threshold_percentile)
    #         print(gc.shape)
    #         print(indices_to_keep.shape)
            mask = np.zeros((img.shape[1], img.shape[2]), dtype=bool)
            mask[gc[:,0],gc[:,1]] = indices_to_keep
        except ValueError as e:
            roi_indices = np.unique(rois)
            roi_indices = roi_indices[roi_indices != 0]
            mask = rois == roi_indices[region_idx]

        selem = np.ones((3,3), dtype=bool)
        mask = morphology.binary_opening(mask, selem)
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

def get_bbox_images(img, mask, padding = 0):
    """ Get cropped images defined by the bounding boxes of ROIS provided by a mask
    """
    bboxes = [p["bbox"] for p in measure.regionprops(mask)]
    cropped_images = []
    for bbox in bboxes:
        r1 = max(bbox[0]-padding, 0)
        c1 = max(bbox[1]-padding, 0)
        r2 = min(bbox[2]+padding, img.shape[1])
        c2 = min(bbox[3]+padding, img.shape[2])
        cropped_images.append(img[:, r1:r2, c1:c2])
    return cropped_images

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

def spike_mask_to_stim_index(spike_mask, pos="start"):
    """ Convert detected spikes from crosstalk into a end of stimulation index for spike-triggered averaging

    """
    diff_mask = np.diff(spike_mask.astype(int))
    if pos == "start":
        stims = np.argwhere(diff_mask==1).ravel()
    elif pos == "end":
        stims = np.argwhere(diff_mask==-1).ravel()+1
    return stims

def image_to_roi_traces(img, label_mask):
    """ Get traces from image using defined roi mask
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]
    traces = [image_to_trace(img, label_mask==l) for l in labels]
    return np.array(traces)

def image_to_trace(img, mask = None):
    """ Convert part of an image to a trace according to a mask
    
    """
    if mask is None:
        trace = img.mean(axis=(1,2))
    elif len(mask.shape)==3:
        masked_img = np.ma.masked_array(img, mask=~mask)
        trace = masked_img.mean(axis=(1,2))
    else:
        pixel_traces = img[:, mask]
        trace = pixel_traces.mean(axis=1)
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

def spline_fit_single_trace(trace, s, knots, plot=False, n_iterations=100, eps=0.01):
    """ Least squares spline fitting of a single timeseries
    """
    x = np.arange(len(trace))
    trace[np.isnan(trace)] = np.min(trace)
    (t,c,k), res,_,_ = interpolate.splrep(x,trace, s=s, task=-1,t=knots, full_output=True,k=3)
    spl = interpolate.BSpline(t,c,k)
    spl_values = spl(x)    
    if plot:
        fig1, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(trace)
        ax1.plot(spl(x))
    
    dspl = spl.derivative()
    d2spl = spl.derivative(nu=2)
    
    def custom_newton_lsq(x, y, bounds = (-np.inf, np.inf)):
        """ solve for spline being a particular value, i.e. (spl(x) - y)**2 = 0
        """
        fx = spl(x)**2 - 2*y*spl(x) + y**2
        dfx = 2*dspl(x)*spl(x) - 2*y*dspl(x)
        x1 = x - fx/dfx
        x1 = max(bounds[0], x1)
        x1 = min(bounds[1], x1)
        return x1
    
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
        return beta, spl
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
    # min_val = spl(last_min_before_max)
    min_val = np.percentile(spl_values[:global_max], 10)
    halfmax_magnitude = (global_max_val - min_val)/2 + min_val
    try:
#         halfmax = optimize.minimize_scalar(lambda x: (spl(x) - halfmax_magnitude)**2, method='bounded', \
#                                        bounds=[last_min_before_max, global_max]).x
        try:
            # hm = np.argmin((spl_values[last_min_before_max:global_max]-\
                               # halfmax_magnitude)**2) + last_min_before_max
            reversed_values = np.flip(spl_values[:global_max])
            moving_away_from_hm = np.diff(reversed_values-halfmax_magnitude)**2 > 0
            close_to_hm = (reversed_values-halfmax_magnitude)**2 < ((global_max_val - min_val)*5/global_max)**2
            hm = global_max - np.argwhere(close_to_hm[:-1] & moving_away_from_hm).ravel()[0]
            # hm = np.argmin((spl_values[:global_max]-\
                               # halfmax_magnitude)**2)
#             halfmax = optimize.minimize_scalar(lambda x: (spl(x) - halfmax_magnitude)**2, method='bounded', \
#                                            bounds=[hm-1, hm+1]).x
            hm0 = hm
            for j in range(n_iterations):
                hm1 = custom_newton_lsq(hm0, halfmax_magnitude, bounds=[hm-1, hm+1])
                if (hm1 - hm0)**2 < eps**2:
                    break
                hm0 = hm1
            halfmax = hm1
        except Exception as e:
            print(e)
            beta = np.nan*np.ones(4)
            return beta, spl
        if plot:
            ax1.plot(halfmax, spl(halfmax), "gx")
    except Exception as e:
        print(extrema)
        print(d2)
        print(last_min_before_max)
        print(global_max)
        raise e
    beta = np.array([global_max, halfmax, global_max_val/min_val,res])
    return beta, spl
    
def spline_timing(img, s=0.1, n_knots=4, upsample_rate=1):
    """ Perform spline fitting to functional imaging data do determine wavefront timing
    """
    knots = np.linspace(0, img.shape[0]-1, num=n_knots)[1:-1]
    q = np.apply_along_axis(lambda tr: spline_fit_single_trace(tr, s, knots), 0, img)
    beta = np.moveaxis(np.array(list(q[0].ravel())).reshape((img.shape[1], img.shape[2],-1)), 2, 0)
    x = np.arange(img.shape[0]*upsample_rate)/upsample_rate
    smoothed_vid = np.array([spl(x) for spl in q[1].ravel()])
    smoothed_vid = np.moveaxis(smoothed_vid.reshape((img.shape[1], img.shape[2],-1)), 2, 0)
    return beta, smoothed_vid

def process_isochrones(beta, dt, pct_threshold=50, plot=False):
    """ Clean up spline fitting to better visualize isochrones: get rid of nans and low-amplitude values 
    """
    amplitude = beta[2,:,:]
    amplitude_nanr = np.copy(amplitude)
    minval = np.nanmin(amplitude)
    amplitude_nanr[np.isnan(amplitude)] = minval
    # thresh = (np.percentile(amplitude_nanr,90)-1)*pct_threshold/100+1
    thresh = np.percentile(amplitude_nanr, pct_threshold)
    print(thresh)
    # thresh = filters.threshold_otsu(amplitude_nanr)
    mask = amplitude_nanr>thresh
    mask = morphology.binary_opening(mask, selem=morphology.disk(2))
    mask = morphology.binary_closing(mask, selem=morphology.disk(2))
    
    labels = measure.label(mask)
    label_values, counts = np.unique(labels, return_counts=True)
    label_values = label_values[1:]
    counts = counts[1:]
    mask = labels == label_values[np.argmax(counts)]
    
    if plot:
        fig1, ax1 = plt.subplots(figsize=(3,3))
        visualize.display_roi_overlay(amplitude_nanr, mask.astype(int), ax = ax1)
    
    hm = beta[1,:,:]
    average_regional_nans = ndimage.convolve(np.isnan(hm), np.ones((3,3)))
    convinput = np.copy(hm)
    convinput[np.isnan(hm)] = 0
    kernel = np.ones((3,3))
    kernel[1,1] = 0
    hm_nans_removed = np.copy(hm)
    hm_nans_removed[np.isnan(hm)] = ndimage.convolve(convinput,kernel)[np.isnan(hm)]
    
    hm_smoothed = ndimage.median_filter(hm_nans_removed, size=3)
    hm_smoothed = ndimage.gaussian_filter(hm_smoothed, sigma=1)
    hm_nan = np.copy(hm_smoothed)
    hm_nan[~mask] = np.nan
    hm_nan = hm_nan*dt*1000
    
    return hm_nan

def estimate_local_velocity(activation_times, deltax=7,deltay=7,deltat=100):
    """ Use local polynomial fitting strategy from Bayley et al. 1998 to determine velocity from activation map
    """
    X, Y = np.meshgrid(np.arange(activation_times.shape[0]), np.arange(activation_times.shape[1]))
    coords = np.array([Y.ravel(), X.ravel()]).T
    Tsmoothed = np.ones_like(activation_times)*np.nan
    residuals = np.ones_like(activation_times)*np.nan
    v = np.ones((2, activation_times.shape[0], activation_times.shape[1]))*np.nan
    for y,x in coords:
        # if np.isnan(activation_times[y, x]):
        #     continue
        local_X, local_Y = np.meshgrid(np.arange(max(0, x-deltax), min(activation_times.shape[1], x+deltax+1)),\
                                       np.arange(max(0, y-deltay), min(activation_times.shape[0], y+deltay+1)))\
                            - np.array([x,y])[:,np.newaxis,np.newaxis]
        local_times = activation_times[max(0, y-deltay):min(activation_times.shape[0], y+deltay+1),\
                                      max(0, x-deltax):min(activation_times.shape[1], x+deltax+1)]
        local_X = local_X.ravel()
        local_Y = local_Y.ravel()
        A = np.array([local_X**2, local_Y**2, local_X*local_Y, local_X, local_Y, np.ones_like(local_X)]).T
        b = local_times.ravel() 
        A = A[~np.isnan(b),:]
        b = b[~np.isnan(b)]
        mask = np.abs(b - activation_times[y,x]) < deltat
        A = A[mask,:]
        b = b[mask]
        
        if len(b) < 10:
            continue
        try:
            p, res, _, _ = np.linalg.lstsq(A, b)
            Tsmoothed[y,x] = p[5]
            residuals[y,x] = res
            v[:,y,x] = p[3:5]/(p[3]**2 + p[4]**2)
        except Exception as e:
            pass
    return v, Tsmoothed

def correct_photobleach(img, mask=None, method="localmin", nsamps=51, invert=False):
    """ Perform photobleach correction on each pixel in an image
    """
    if method == "linear":
        # Perform linear fit to each pixel independently over time
        corrected_img = np.zeros_like(img)
        for y in range(img.shape[1]):
            for x in range(img.shape[2]):
                corrected_trace, _ = traces.correct_photobleach(img[:,y,x])
                corrected_img[:, y,x] = corrected_trace
                
    elif method == "localmin":
        # 
        if invert:
            mean_img = img.mean(axis=0)
            raw_img = 2*mean_img - img
        else:
            raw_img = img
        mean_trace = image_to_trace(raw_img, mask)
        _, pbleach = traces.correct_photobleach(mean_trace, method=method, nsamps=nsamps)
        corrected_img = np.divide(raw_img, pbleach[:,np.newaxis,np.newaxis])
        print(corrected_img.shape)
        
    elif method == "monoexp":
        mean_trace = image_to_trace(img, mask)
        tpoints = np.arange(len(mean_trace))
        def expon(x,a,k, c):
            y = a*np.exp(x*k) + c
            return(y)
        guess_tc = -(np.percentile(mean_trace, 95)/np.percentile(mean_trace,5))/len(mean_trace)
        p0 = (np.max(mean_trace)-np.min(mean_trace), guess_tc, np.min(mean_trace))
        popt, _ = optimize.curve_fit(expon, tpoints, mean_trace, p0=p0, bounds=([0,-np.inf,0], \
                                                                                [np.inf,0,np.inf]))
        fig1, ax1 = plt.subplots(figsize=(6,6))
        ax1.scatter(tpoints,mean_trace)
        ax1.plot(tpoints, popt[0]*np.exp(popt[1]*tpoints) + popt[2], color="red")
        
        corrected_img = img/((popt[2]+popt[0]*np.exp(tpoints*popt[1]))[:,np.newaxis,np.newaxis]) + popt[2]
    elif method == "decorrelate":
        # Zero the mean over time
        mean_img = img.mean(axis=0)
        t_zeroed = img - mean_img
        data_matrix = t_zeroed.reshape((t_zeroed.shape[0], -1))

        # correlate to mean trace
        mean_trace = data_matrix.mean(axis=1)
        corr = np.matmul(data_matrix.T, mean_trace)/np.dot(mean_trace, mean_trace)
        resids = data_matrix - np.outer(mean_trace, corr)
        
        corrected_img = mean_img + resids.reshape(img.shape)
    else:
        raise ValueError("Not Implemented")
        
    return corrected_img

def get_image_dFF(img, baseline_percentile=10):
    """ Convert a raw image into dF/F

    """
    baseline = np.percentile(img, baseline_percentile, axis=0)
    dFF = img/baseline
    return dFF


def get_spike_videos(img, peak_indices, before, after, normalize_height=True):
    """ Generate spike-triggered videos of a defined length from a long video and peak indices

    """
    spike_imgs = np.ones((len(peak_indices), before+after, img.shape[1], img.shape[2]))*np.nan
    for pk_idx, pk in enumerate(peak_indices):
        before_pad_length = max(before - pk, 0)
        after_pad_length = max(0, pk+after - img.shape[0])
        spike_img = np.concatenate([np.ones((before_pad_length,img.shape[1],img.shape[2]))*np.nan,
                                                img[max(0, pk-before):min(img.shape[0],pk+after),:,:],
                                        np.ones((after_pad_length,img.shape[1],img.shape[2]))*np.nan])
        if normalize_height:
            spike_img /= np.nanmax(spike_img)
        spike_imgs[pk_idx,:,:,:] = spike_img
    return spike_imgs

def spike_triggered_average_video(img, peak_indices, before, after, include_mask=None, normalize_height=False, full_output=False):
    """ Create a spike-triggered average video
    """
    if include_mask is None:
        include_mask = np.ones_like(peak_indices, dtype=bool)
    spike_triggered_images = get_spike_videos(img,peak_indices[include_mask], before, after, normalize_height=normalize_height)
    sta = np.nanmean(spike_triggered_images, axis=0)
    if full_output:
        return sta, spike_triggered_images
    else:
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

def downsample_video(raw_img, downsample_factor, aa="gaussian"):
    """ Downsample video in space with optional anti-aliasing
    """
    if downsample_factor == 1:
        di = raw_img
    else:
        if aa == "butter":
            sos = signal.butter(4, 1/downsample_factor, output='sos')
            smoothed = np.apply_over_axes(lambda a, axis: np.apply_along_axis(lambda x: signal.sosfiltfilt(sos, x), axis, a), raw_img, [1,2])
            di = smoothed[:,np.arange(smoothed.shape[1], step=downsample_factor, dtype=int),:]
            di = di[:,:,np.arange(smoothed.shape[2], step=downsample_factor, dtype=int)]            
        elif aa == "gaussian":
            smoothed = ndimage.gaussian_filter(raw_img, [0,downsample_factor,downsample_factor])
            di = smoothed[:,np.arange(smoothed.shape[1], step=downsample_factor, dtype=int),:]
            di = di[:,:,np.arange(smoothed.shape[2], step=downsample_factor, dtype=int)]
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
    
    
def image_to_sta(raw_img, downsample_factor=1, fs=1, mask=None, plot=False, savedir=None, prom_pct=90, sta_bounds="auto", aa=True, dff_time_downsample=1, exclude_pks = None, offset=0):
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
        if dff_time_downsample > 1:
            skio.imsave(os.path.join(savedir, "dFF.tif"), transform.downscale_local_mean(raw_img, (dff_time_downsample,1,1)))
        else:
            skio.imsave(os.path.join(savedir, "dFF.tif"), dFF_img)
        skio.imsave(os.path.join(savedir, "pca.tif"), pca_img)
        np.savez(os.path.join(savedir, "dFF_mean.npz"), dFF_mean=dFF_mean)
    
    # Identify spikes
    ps = np.percentile(dFF_mean,[10,prom_pct])
    prominence = ps[1] - ps[0]
    pks, _ = signal.find_peaks(dFF_mean, prominence=prominence)
    
    if len(pks) == 0:
        return None
    
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

# def identify_hearts(img, expected_embryos, prev_coms=None, prev_mask_labels=None, fill_missing=True, band_bounds=(0.1, 1), \
#                    band_cutoff=0.4, full_output=False, opening_size=5, dilation_size=15, f_s=1):
#     """ Pick out hearts from widefield experiments using PCA and power content in frequency band
#     """
    
#     mean_img = img.mean(axis=0)
#     std_img = np.std(img, axis=0)
#     cv_img = std_img/mean_img
#     intensity_mask = mean_img > np.percentile(mean_img, 75)
#     cv_img[~intensity_mask] = 0
    
#     cv_mask = cv_img > np.percentile(cv_img[intensity_mask],85)
    
#     xx = morphology.binary_opening(cv_mask, selem= np.ones((opening_size,opening_size)))
#     xxx = morphology.binary_dilation(xx, selem= np.ones((dilation_size,dilation_size)))
#     labelled = measure.label(xxx)
#     rd, gc = get_all_region_data(img, labelled)
    
#     new_mask = np.zeros_like(mean_img, dtype=bool)
    
#     for i, regiondata in enumerate(rd):
#         pca = PCA(n_components=5)
#         rd_norm = regiondata-np.mean(regiondata, axis=0)
#         rd_norm = rd_norm/np.max(np.abs(rd_norm), axis=0)
#         pca.fit(rd_norm)
#         for c in pca.components_:
#             trace = np.matmul(rd_norm, c)
#             N = len(trace)
#             yf = fft(trace)
#             xf = fftfreq(len(trace), 1/f_s)[:N//2]
#             abs_power = np.abs(yf[0:N//2])
#             norm_abs_power = abs_power/np.sum(abs_power)
#             band_power = np.sum(norm_abs_power[(xf>band_bounds[0]) & (xf<band_bounds[1])])
#             if band_power > band_cutoff:
#                 comp_abs = np.abs(c)
#                 correct_indices = comp_abs > filters.threshold_otsu(comp_abs)
#                 mask_coords = gc[i][correct_indices]
#                 mask_coords = tuple(zip(*mask_coords.tolist()))
#                 new_mask[mask_coords] = 1
#                 break
#     new_mask = morphology.binary_opening(new_mask, selem=np.ones((opening_size,opening_size)))
#     new_mask = morphology.binary_dilation(new_mask, selem=np.ones((dilation_size,dilation_size)))
#     closing_size = ((int(dilation_size*1.5))//2)*2+1
#     new_mask = morphology.binary_closing(new_mask, selem=np.ones((closing_size,closing_size)))
    
#     new_mask_labels = measure.label(new_mask)
# #     fig1, axes = plt.subplots(1,3, figsize=(12,4))
# #     axes[0].imshow(new_mask_labels)

#     coms = ndi.center_of_mass(new_mask, labels=new_mask_labels, index=np.arange(1,np.max(new_mask_labels)+1))
#     coms = np.array(coms)
    
    
    
# #     print(len(coms))
#     if len(coms) > expected_embryos:
#         plt.imshow(new_mask_labels)
#         print("Extra segments found, sorting by band power")
        
#         band_powers = []
#         for roi in range(1, np.max(new_mask_labels)+1):
#             trace = image_to_trace(img, \
#                                           mask=np.tile(new_mask_labels==roi, \
#                                                        (img.shape[0],1,1)))
#             yf = fft(trace -np.mean(trace))
#             xf = fftfreq(len(trace), 1/f_s)[:N//2]
#             abs_power = np.abs(yf[0:N//2])
#             norm_abs_power = abs_power/np.sum(abs_power)
#             band_power = np.sum(norm_abs_power[(xf>band_bounds[0]) & (xf<band_bounds[1])])
#             band_powers.append(band_power)
            

#         indices_to_keep = np.argsort(-np.array(band_powers))[:expected_embryos]
#         coms = coms[indices_to_keep,:]
#         ml_temp = np.zeros_like(new_mask_labels)
#         for i, j in enumerate(indices_to_keep):
#             ml_temp[new_mask_labels==j+1] = i+1
#         new_mask_labels = ml_temp
        
#     if prev_coms is not None:
# #         print("Detected COMs: %d" % coms.shape[0])
# #         print("Previous frame COMs: %d" % prev_coms.shape[0])
#         new_coms_ordered = {}
#         n_new_rois = 0
#         new_mask_copy = np.zeros_like(new_mask_labels, dtype=np.uint8)
        
#         unassigned_coms = []
#         # Try to link ROIs segmented from this image to ROIs from previous image so index is maintained
#         for idx in range(coms.shape[0]):
#             com = coms[idx,:]
#             dist = np.sum(np.power(com - prev_coms,2), axis=1)
#             min_idx = np.argmin(dist)
#             if np.sqrt(dist[min_idx]) < 25:
# #                 print(idx)
#                 new_mask_copy[new_mask_labels==(idx+1)] = min_idx+1
#                 if min_idx in new_coms_ordered:
#                     new_coms_ordered[min_idx] = np.mean([com, new_coms_ordered[min_idx]])
#                 else:
#                     new_coms_ordered[min_idx] = com
#             # If a ROI was not found in the neighborhood of one from the previous image, add a new value
#             elif prev_coms.shape[0] + n_new_rois < expected_embryos:
#                 n_new_rois += 1
#                 new_label_val = prev_coms.shape[0] + n_new_rois
# #                 print("New label found %d" % new_label_val)
#                 new_mask_copy[new_mask_labels==(idx+1)] = new_label_val
#                 new_coms_ordered[new_label_val-1] = com
#             else:
#                 unassigned_coms.append(com)
        
#         # If any ROIs from the previous image were missing, fill them in
#         if fill_missing:
#             new_labels = set(np.unique(new_mask_copy).tolist())
#             old_labels = set(np.unique(prev_mask_labels).tolist())
            
#             missing_old_labels = old_labels - new_labels
# #             print("Missing old labels ", missing_old_labels)
            
#             for ol in missing_old_labels:
#                 new_mask_copy[prev_mask_labels==ol] = ol
#                 new_coms_ordered[ol-1] = prev_coms[ol-1,:]
#         new_mask_labels = new_mask_copy
#         coms = np.zeros((len(new_coms_ordered), 2))
# #         print("New COMs length: %d" % len(new_coms_ordered))
# #         all_indices_assigned = []
#         for idx, com in new_coms_ordered.items():
#             coms[idx, :] = com
# #             all_indices_assigned.append(idx)
# #         print(sorted(all_indices_assigned))
# #         axes[1].imshow(new_mask_labels)
# #         axes[2].imshow(new_mask_labels-prev_mask_labels)
# #         plt.tight_layout()
#     print(coms.shape)
#     if full_output:
#         return new_mask_labels, coms, intensity_mask, cv_mask, labelled
#     else:
#         return new_mask_labels, coms



def identify_hearts(img, prev_coms=None, prev_mask_labels=None, fill_missing=True, band_bounds=(0.1, 2), \
                   band_threshold=0.45, full_output=False, opening_size=5, dilation_size=15, f_s=1, \
                     intensity_threshold=0.5, bbox_offset=5, corr_threshold=0.9):
    """ Pick out hearts from widefield experiments using PCA and power content in frequency band
    """
#     print("band_bounds:", band_bounds)
#     print("band_threshold:", band_threshold)
#     print("opening_size:", opening_size)
#     print("dilation_size:", dilation_size)
#     print("f_s:", f_s)
    mean_img = img.mean(axis=0)
    zeroed_image = img-mean_img
    intensity_mask = mean_img > np.percentile(mean_img, intensity_threshold*100)
    pixelwise_fft = fft(zeroed_image, axis=0)

    N_samps = img.shape[0]
    
    fft_freq = fftfreq(N_samps, 1/f_s)[:N_samps//2]

    abs_power = np.abs(pixelwise_fft[:N_samps//2,:,:])**2
    norm_abs_power = abs_power/np.sum(abs_power, axis=0)

    band_power = np.sum(norm_abs_power[(fft_freq>band_bounds[0]) & (fft_freq<band_bounds[1]),:,:], axis=0)
    smoothed_band_power = filters.median(band_power, selem=np.ones((5,5)))*intensity_mask.astype(int)
    processed_band_power = morphology.binary_opening((smoothed_band_power > band_threshold), selem=np.ones((3,3)))
    initial_guesses = measure.label(processed_band_power)
    bboxes = [p["bbox"] for p in measure.regionprops(initial_guesses)]
    new_mask = np.zeros_like(mean_img, dtype=bool)
#     new_mask_labels = np.deepcopy(initial_guesses)
    n_rois = np.max(initial_guesses)
    corr_img = np.zeros((n_rois, mean_img.shape[0], mean_img.shape[1]), dtype=float)

    
    for i in range(1, n_rois+1):
        bbox = bboxes[i-1]
        r1 = max(bbox[0]-bbox_offset, 0)
        c1 = max(bbox[1]-bbox_offset, 0)
        r2 = min(bbox[2]+bbox_offset, initial_guesses.shape[0])
        c2 = min(bbox[3]+bbox_offset, initial_guesses.shape[1])
#         print(r1,r2,c1,c2)
        roi_mask = np.zeros_like(initial_guesses, dtype=bool)
        roi_mask[r1:r2, c1:c2] = 1    
    
        initial_trace = image_to_trace(zeroed_image, mask = roi_mask)
        

        roi_traces = zeroed_image[:,roi_mask]
#         print(roi_traces.shape)
        corrs = np.apply_along_axis(lambda x: stats.pearsonr(initial_trace, x)[0], 0, roi_traces)
        corrs = corrs.reshape((r2-r1, c2-c1))
        corr_img[i-1, r1:r2, c1:c2] = corrs
        
    corr_mask = morphology.binary_opening(np.max(corr_img>corr_threshold, axis=0), selem=np.ones((opening_size,opening_size)))
#     remaining_rois = np.unique(initial_guesses[corr_mask])
#     print(remaining_rois)
#     remaining_rois = remaining_rois[np.nonzero(remaining_rois)]
#     for i in remaining_rois:
#         new_mask[initial_guesses==i] = 1
#     print(dilation_size)
    new_mask = morphology.binary_dilation(corr_mask, selem=np.ones((dilation_size, dilation_size)))
    
    new_mask_labels = measure.label(new_mask)

    coms = ndi.center_of_mass(new_mask, labels=new_mask_labels, index=np.arange(1,np.max(new_mask_labels)+1))
    coms = np.array(coms)

    print(coms.shape)
    if full_output:
        return new_mask_labels, coms, intensity_mask, abs_power, smoothed_band_power, initial_guesses, corr_img
    else:
        return new_mask_labels, coms

def segment_widefield_series(filepaths, expected_embryos, downsample_factor=1, remove_from_start=0, remove_from_end=0, opening_size=3, dilation_size=3, band_bounds=(0.1,2), f_s=1, band_threshold=0.45, intensity_threshold=0.5, corr_threshold=0.9):
    """ Run heart segmentation for widefield experiments on all files in a folder.
    Expected in true chronological order (not reverse).
    """
    curr_labels = None
    curr_coms = None
    raw = None
    frames = []
    exclude_from_write = np.zeros(len(filepaths), dtype=bool)
    img_shape = None
    # Perform inital segmentation
    for idx, f in enumerate(np.flip(filepaths)):
        try:
            del raw
        except Exception as e:
            pass
        print(f)
        try:
            raw = skio.imread(f)
            if img_shape is None:
                img_shape = raw.shape[1:3]
            else:
                if raw.shape[1:3] != img_shape:
                    raise Exception("Incorrect image shape")
            raw = raw[remove_from_start:raw.shape[0]-remove_from_end,:,:]
        except Exception as e:
            print(e)
            exclude_from_write[idx] = 1
            continue
        if downsample_factor > 1:
            downsample = transform.downscale_local_mean(raw, (1,downsample_factor,downsample_factor))
        else:
            downsample = raw
        try:
            curr_labels, curr_coms = identify_hearts(downsample, \
                        prev_coms=curr_coms, prev_mask_labels=curr_labels, \
                        opening_size=opening_size, dilation_size=dilation_size, band_threshold=band_threshold, \
                                                     intensity_threshold=intensity_threshold, \
                                                 band_bounds=band_bounds, f_s=f_s, corr_threshold=corr_threshold)
        except Exception as e:
            print(e)
            print(curr_coms)
            pass
        frames.append(np.copy(curr_labels))
        
    vid = np.array(frames)
    return vid, np.flip(exclude_from_write)

def pairwise_mindist(x, y):
    """ Calculate minimum distance between each point in the list x and each point in the list y
    """
    
    pairwise_dist = np.abs(np.subtract.outer(x, y))
    mindist_indices = np.argmin(pairwise_dist, axis=1)
    mindist = pairwise_dist[np.arange(x.shape[0]), mindist_indices]
    
    return mindist, mindist_indices

def link_frames(curr_labels, prev_labels, prev_coms, radius=15, propagate_old_labels=True):
    """ Connect two ROI segmentations adjacent in time
    """
    curr_mask = curr_labels > 0
    all_curr_labels = np.unique(curr_labels)[1:]
    all_prev_labels = np.unique(prev_labels)[1:]

    curr_coms = ndi.center_of_mass(curr_mask, labels=curr_labels, index=all_curr_labels)
    curr_coms = np.array(curr_coms)
    if len(curr_coms.shape) == 2 and len(prev_coms) > 0:
        curr_coms = curr_coms[:,0] + 1j*curr_coms[:,1]

        mindist, mindist_indices = pairwise_mindist(curr_coms, prev_coms)
        link_curr = np.argwhere(mindist < radius).ravel()

        link_prev = set(mindist_indices[link_curr]+1)

        link_curr = set(link_curr+1)
    elif len(curr_coms.shape) == 1 or len(prev_coms)==0:
        link_curr = set([])
        link_prev = set([])
    all_curr_labels_set = set(all_curr_labels)
    all_prev_labels_set = set(all_prev_labels)

    new_labels = np.zeros_like(curr_labels)
    
    for label in link_curr:
        new_labels[curr_labels == label] = all_prev_labels[mindist_indices[label-1]]
    
    if propagate_old_labels:
        unassigned_prev_labels = all_prev_labels_set - link_prev
        for label in unassigned_prev_labels:
            new_labels[prev_labels == label] = label
    
    unassigned_curr_labels = all_curr_labels_set - link_curr
    new_rois_counter = 0
    try:
        starting_idx = np.max(all_prev_labels)+1
    except Exception:
        starting_idx = 1
    for label in unassigned_curr_labels:
        if np.all(new_labels[curr_labels==label]==0):
            new_labels[curr_labels == label] = starting_idx + new_rois_counter
            new_rois_counter += 1
    new_mask = new_labels > 0
    new_coms =  ndi.center_of_mass(new_mask, labels=new_labels, index=np.unique(new_labels)[1:])
    new_coms = np.array(new_coms)
    new_coms = new_coms[:,0] + 1j*new_coms[:,1]
    return new_labels, new_coms

def link_stack(stack, step=-1, radius=15, propagate_old_labels=True):
    if step <0:
        curr_t = 1
    else:
        curr_t = stack.shape[0]-1

    prev_labels = stack[curr_t+step]
    prev_mask = prev_labels > 0
    
    prev_coms = ndi.center_of_mass(prev_mask, labels=prev_labels, index=np.arange(1,np.max(prev_labels)+1))
    prev_coms = np.array(prev_coms)
    if len(prev_coms.shape)==2:
        prev_coms = prev_coms[:,0] + 1j*prev_coms[:,1]
    new_labels = [prev_labels]
    while curr_t >=0 and curr_t < stack.shape[0]:
        curr_labels = stack[curr_t]
        curr_labels, curr_coms = link_frames(curr_labels, prev_labels, prev_coms,\
             radius=radius, propagate_old_labels=propagate_old_labels)
        prev_labels = curr_labels
        prev_coms = curr_coms
        curr_t -= step
        new_labels.append(curr_labels)
    new_labels = np.array(new_labels)
    return new_labels

def filter_by_appearances(linked_vid, unlinked_vid, threshold=1/3):
    roi_found = []
    for roi in np.arange(1, np.max(linked_vid)+1):
        roi_linked = linked_vid==roi
        found_in_unlinked = []
        for i in range(roi_linked.shape[0]):
            detected = unlinked_vid[i][roi_linked[i]]
            found_in_unlinked.append(np.any(detected>0)*(len(detected) > 0))
        roi_found.append(found_in_unlinked)
    roi_found = np.array(roi_found)
    keep = np.argwhere(np.sum(roi_found, axis=1)> threshold*linked_vid.shape[0]).ravel()+1
    
    filtered_vid = np.zeros_like(linked_vid)
    for idx, roi in enumerate(keep):
        filtered_vid[linked_vid==roi] = idx+1
    
    return filtered_vid

def closest_non_zero(arr):
    nonzeros = np.argwhere(arr!=0)
#     print(nonzeros)
#     print(nonzeros.shape)
    distances = np.abs(np.subtract.outer(np.arange(len(arr), dtype=int), nonzeros)).reshape((len(arr), -1))
#     print(distances.shape)
    min_indices = np.argmin(distances, axis=1)
    return nonzeros[min_indices]

def fill_missing(vid, threshold=100):
    """ Detect when an ROI drops out of a sequence of movies
    """
    roi_sizes = []
    for roi in range(1, np.max(vid)+1):
        roi_sizes.append(np.sum(vid==roi, axis=(1,2)))
    roi_sizes = np.array(roi_sizes)
    below_threshold = roi_sizes < threshold
    closest_above_threshold = np.apply_along_axis(closest_non_zero, 1, ~below_threshold).squeeze()
    filled_vid = np.zeros_like(vid)
    for i in range(1, closest_above_threshold.shape[0]+1):
        replaced_vals = vid[closest_above_threshold[i-1],:,:] == i
        filled_vid[replaced_vals] = i
    return filled_vid

def get_regularized_mask(img, rois):
    """ Correct noisy segmentation using the fact that the hearts are roughly the same size. Take the mean area and draw a circle with the same area centered at the centroid of each original ROI.
    """
    mask_labels = np.unique(rois)
    mask_labels = mask_labels[mask_labels != 0]

    rps = measure.regionprops(rois)

    areas = [p["area"] for p in rps]
    centroids = np.array([p["centroid"] for p in rps])
    mean_area = np.mean(areas)
    radius = (mean_area/np.pi)**0.5
    
    area_regularized_mask = np.zeros_like(refined_mask)
    for i in range(len(region_masks)):
        rr, cc = draw.disk(tuple(centroids[i,:]), radius)
        valid_indices = (rr < refined_mask.shape[0]) & (cc < refined_mask.shape[1]) & (rr >= 0) & (cc >= 0)
        rr = rr[valid_indices]
        cc = cc[valid_indices]
        area_regularized_mask[rr, cc] = mask_labels[i]
    
    return area_regularized_mask

