import numpy as np
import skimage.io as skio
from typing import Tuple, Union, Callable
from numpy import typing as npt

from scipy import signal, stats, interpolate, optimize, ndimage
import matplotlib.pyplot as plt
from skimage import transform, filters, morphology, measure, draw, exposure, segmentation, feature
from skimage.util import map_array
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn import cluster
import os
import mat73
import warnings
import colorcet as cc
import seaborn as sns
import pickle
import pandas as pd
from cycler import cycler

from . import traces
from . import stats as sstats
from .. import utils
from ..ui import visualize


def regress_video(img: npt.NDArray, trace_array: npt.NDArray,\
                   regress_dc: bool = True) -> npt.NDArray:
    """ Linearly regress arbitrary traces from a video.

    Args:
        img (numpy.ndarray): 3D array of video data (time, x, y).
        trace_array (numpy.ndarray): 2D array of traces to regress (time, traces).
        regress_dc (bool): if True, will regress out the mean of the traces.
    Returns:
        regressed_video (numpy.ndarray): 3D array of video data with traces
                                        regressed out (time, x, y).
    """
    data_matrix = img.reshape((img.shape[0], -1))
    regressed_video = sstats.multi_regress(
        data_matrix, trace_array, regress_dc=regress_dc).reshape(img.shape)
    return regressed_video


def load_dmd_target(rootdir: str, expt_name: str,\
                     downsample_factor: float = 1) -> npt.NDArray[np.bool_]:
    """ Load the DMD target image from the experiment metadata. 
    This is the image that the DMD is trying to project onto the screen.

    Args:
        rootdir (str): root directory of the experiment.
        expt_name (str): name of the experiment (single video).
        downsample_factor (float): factor by which to downsample the image. This deals with the fact that we downsample our images during 
        processing.
    Returns:
        dmd_target (numpy.ndarray[bool]): DMD target mask in image space.
    """
    # Load the .mat file containing the metadata.
    expt_data = mat73.loadmat(os.path.join(
        rootdir, expt_name, "output_data_py.mat"))["dd_compat_py"]
    width = int(expt_data["camera"]["roi"][1])
    height = int(expt_data["camera"]["roi"][3])
    dmd_target = expt_data["dmd_lightcrafter"]['target_image_space']

    # DMD transformation to image space is not perfect, so we need to crop the image to get rid of the black border
    offset_x = int((dmd_target.shape[1] - width)/2)
    offset_y = int((dmd_target.shape[0] - height)/2)
    dmd_target = dmd_target[offset_y:-offset_y, offset_x:-offset_x]
    dmd_target = dmd_target[::downsample_factor,
                            ::downsample_factor].astype(bool)
    return dmd_target


def load_image(rootdir, expt_name, subfolder="", raw=True):
    """ General image loading function that handles various file formats floating around in
    Cohen lab.  

    If raw is True, will try to load the raw data from the .bin file.  Otherwise,
    will try to load the tif files.  If there are multiple tif files, will concatenate them.

    Args:
        rootdir (str): root directory of the experiment.
        expt_name (str): name of the experiment (single video).
        subfolder (str): subfolder of the experiment (different stages of processing).
        raw (bool): if True, will try to load the raw data from the .bin file.  Otherwise, will try to load the tif files.
    Returns:
        imgs (list[numpy.ndarray]): image data.  If there is only one camera, this will be a 3D array (time, x, y).  If there are multiple cameras, this will be a list of 3D arrays.

    """
    d = os.path.join(rootdir, subfolder)
    all_files = os.listdir(d)
    expt_files = 0
    # Load the .mat file containing the metadata
    expt_data = utils.load_experiment_metadata(rootdir, expt_name)
    imgs = []

    if raw:
        try:
            # Load binary images for each camera
            for i in range(len(expt_data["cameras"])):
                width = int(expt_data["cameras"][i]["roi"][1])
                height = int(expt_data["cameras"][i]["roi"][3])
                rawpath = os.path.join(
                    rootdir, subfolder, expt_name, "frames.bin")
                if not os.path.exists(rawpath):
                    rawpath = os.path.join(
                        rootdir, subfolder, expt_name, f"frames{i+1}.bin")
                if not os.path.exists(rawpath):
                    rawpath = os.path.join(
                        rootdir, subfolder, expt_name, "Sq_camera.bin")
                imgs.append(np.fromfile(rawpath, dtype=np.dtype(
                    "<u2")).reshape((-1, height, width)))
        except Exception as ex:
            # If there is an error, try to load the tif files instead
            raw = False
            warnings.warn(
                "Error loading raw data, trying to load tif files instead")
            warnings.warn(str(ex))

    if not raw:
        # Identify the number of tif files.
        for f in all_files:
            if expt_name in f and ".tif" in f:
                expt_files += 1
        if expt_files == 1:
            # If there is only one tif file, load it.
            imgs.append(skio.imread(os.path.join(d, f"{expt_name}.tif")))
        else:
            # If there are multiple tif files, load them and concatenate them.
            img = [skio.imread(os.path.join(
                d, f"{expt_name}_block{block+1}.tif")) for block in range(expt_files)]
            imgs.append(np.concatenate(img, axis=0))
    if len(imgs) == 1:
        imgs = imgs[0]
    if "frame_counter" in expt_data.keys():
        # Check to make sure that the number of frames in the .bin file matches the number of frames expected from the metadata.
        fc_max = np.max(expt_data["frame_counter"])
        for i in range(len(imgs)):
            if fc_max > imgs[i].shape[0]:
                n_frames_dropped = fc_max - imgs[i].shape[0]
                warnings.warn(f"{n_frames_dropped} frames dropped")
                last_tidx = imgs[i].shape[0]
            else:
                last_tidx = int(min(
                    fc_max, expt_data["cameras"][i]["frames_requested"] - expt_data["cameras"][i]["dropped_frames"]))
            imgs[i] = imgs[i][:last_tidx]
    return imgs, expt_data


def load_confocal_image(path, direction="both", extra_offset=2):
    """ Load a confocal image from a .mat file, because it is stored as a DAQ output trace.
    If direction is "fwd", will return the forward scan only.

    Inputs:
        path: path to the .mat file
        direction: "fwd" or "both"
        extra_offset: number of pixels to add to the shifting of reverse scan to align it with forward scan.
            This is a hack to deal with the fact that the reverse scan is shifted by a few pixels relative to the forward scan.
    Returns:
        img: confocal image (z-stack).
    """
    # Load the .mat file containing the metadata and extract information about the scan.
    matdata = mat73.loadmat(path)['dd_compat_py']
    confocal = matdata['confocal']
    points_per_line = int(confocal["points_per_line"])
    numlines = int(confocal["numlines"])

    # Turn output trace into an image
    img = confocal['PMT'].T.reshape(
        (confocal["PMT"].shape[1], numlines, 2, points_per_line))

    # If direction is "fwd", return the forward scan only.
    if direction == "fwd":
        return img[:, :, 0, :]

    # If direction is "both", return the mean of the forward and reverse scans.
    elif direction == "both":
        fwd_scan = img[:, :, 0, :]
        rev_scan = img[:, :, 1, :]

        reshaped_xdata = confocal['xdata'].reshape(numlines, 2,
                                                   points_per_line)
        pixel_step = reshaped_xdata[0, 0, 1] - reshaped_xdata[0, 0, 0]
        offset = np.min(
            np.mean(confocal["galvofbx"][:int(confocal["points_per_line"]), :], axis=1))

        # Shift the reverse scan to align it with the forward scan.
        offset_pix = offset/pixel_step
        offset_idx = int(np.round(-(offset_pix+extra_offset)))
        revscan_shift = np.roll(np.flip(rev_scan, axis=2), offset_idx, axis=2)
        revscan_shift[:, :, :offset_idx] = np.mean(revscan_shift, axis=(1, 2))[
            :, np.newaxis, np.newaxis]

        mean_img = (fwd_scan + revscan_shift)/2
        return mean_img

    else:
        raise Exception("Not implemented")


def generate_invalid_frame_indices(stim_trace):
    """ Identify frames that are invalid due to cross-talk from blue-light stimulation.

    Inputs:
        stim_trace: 1D array of blue light stimulation trace.
    Returns:
        invalid_indices_camera: 1D array of indices of invalid frames.
    """

    invalid_indices_daq = np.argwhere(stim_trace > 0).ravel()
    invalid_indices_camera = np.concatenate(
        (invalid_indices_daq, invalid_indices_daq+1))
    return invalid_indices_camera


def get_all_region_data(img, mask):
    """ Turn all mask regions into pixel-time traces of intensity

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different region.
    Returns:
        region_data: list of 2D arrays (timepoints x pixels) for each region.
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

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different region.
        region_index: integer value of the region to extract.
    Returns:
        region_data: 2D array (timepoints x pixels) for the specified region.

    """
    global_coords = np.argwhere(mask == region_index)
    region_data = np.zeros((img.shape[0], global_coords.shape[0]))
    # Loop over pixels in the region and extract the intensity values
    for px_idx in range(global_coords.shape[0]):
        px = global_coords[px_idx]
        region_data[:, px_idx] = img[:, px[0], px[1]]
    return region_data, global_coords


def generate_cropped_region_image(intensity: npt.NDArray,
    global_coords: Union[npt.NDArray, Tuple[int, int]]) -> npt.NDArray:
    """ Turn an unraveled list of intensities back into an image based on the
    bounding box of the specified global coordinates.

    Inputs:
        intensity (npt.ArrayLike): 1D array of intensity values.
        global_coords (npt.NDArray): defined shape of image or 2D array of
            explicit global coordinates (pixels x 2).
    Returns:
        img (npt.NDArray): 2D array of intensity values.
    """
    if isinstance(global_coords, tuple):
        img = intensity.reshape(global_coords)
    else:
        global_coords_rezeroed = global_coords - np.min(global_coords, axis=0)
        if len(intensity.shape) == 1:
            img = np.zeros(np.max(global_coords_rezeroed, axis=0)+1)
            for idx in range(intensity.shape[0]):
                curr_px = global_coords_rezeroed[idx, :]
                img[curr_px[0], curr_px[1]] = intensity[idx]
        elif len(intensity.shape) == 2:
            img = np.zeros([intensity.shape[0]] +
                           list(np.max(global_coords_rezeroed, axis=0)+1))
            for idx in range(intensity.shape[1]):
                curr_px = global_coords_rezeroed[idx, :]
                img[:, curr_px[0], curr_px[1]] = intensity[:, idx]
        else:
            raise TypeError("Expected 1D or 2D array of intensities")
    return img


def get_bbox_images(img, mask, padding=0):
    """ Get cropped images defined by the bounding boxes of ROIS provided by a mask

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different region.
        padding: number of pixels to add to the bounding box on each side.
    Returns:
        cropped_images: list of 3D arrays (timepoints x pixels x pixels) for the bounding box defined by each region.
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


def plot_pca_data(pca: PCA, raw_data: npt.NDArray, 
                     gc: Union[npt.NDArray,Tuple[int, int]], 
                     n_components: int = 5, 
                     pc_title: Union[Callable[..., str], None] = None):
    """ Show spatial principal components of a video and the corresponding
    temporal trace (dot product).

    Inputs:
        pca (sklearn.decomposition.PCA): sklearn.decomposition.PCA object
        raw_data (npt.NDArray): 2D array of raw data (timepoints x pixels)
        gc (Union[npt.NDArray, Tuple[int, int]]): 2D array of global coordinates
            (pixels x 2) OR tuple of (rows, cols) for the shape of the image.
        n_components (int): number of principal components to display.
        pc_title (Union[Callable, None]): function to generate a title for
            each principal component. If None, the default title is used.
    Returns:
        None
    """
    if pc_title is None:
        pc_title = lambda j, k : f"PC {j+1} (Fraction Var:{pca.explained_variance_ratio_[j]:.3f})"
    for i in range(n_components):
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = axes.ravel()
        comp = pca.components_[i]
        cropped_region_image = generate_cropped_region_image(comp, gc)
        dot_trace = np.matmul(raw_data, comp)
        axes[0].imshow(cropped_region_image)
        axes[0].set_title(pc_title(i, comp))
        axes[1].set_title("PC Value")
        axes[1].plot(dot_trace)


def crosstalk_mask_to_stim_index(crosstalk_mask, pos="start"):
    """ Convert detected spikes from crosstalk into an index for spike-triggered averaging. Use either upward or downward edge.

    Inputs:
        crosstalk_mask: 1D array of boolean values indicating whether a spike was detected.
    Returns:
        stims: 1D array of indices corresponding to the start or end of each spike.
    """
    diff_mask = np.diff(crosstalk_mask.astype(int))
    if pos == "start":
        stims = np.argwhere(diff_mask == 1).ravel()
    elif pos == "end":
        stims = np.argwhere(diff_mask == -1).ravel()+1
    return stims


def image_to_roi_traces(img, label_mask):
    """ Get traces from image using defined ROI mask, where each ROI is defined by a unique integer value.

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        label_mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different region.
    Returns:
        traces: 2D array of traces (ROIs x timepoints)
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]
    traces = [image_to_trace(img, label_mask == l) for l in labels]
    return np.array(traces)


def image_to_trace(img, mask=None):
    """ Average part of an image to a trace according to a binary mask.

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). If None, average over entire image.
    Returns:
        trace: 1D array of trace values (timepoints)
    """
    if mask is None:
        trace = img.mean(axis=(1, 2))
    elif len(mask.shape) == 3:
        masked_img = np.ma.masked_array(img, mask=~mask)
        trace = masked_img.mean(axis=(1, 2))
    else:
        pixel_traces = img[:, mask]
        trace = pixel_traces.mean(axis=1)
    return trace


def interpolate_invalid_values(img, mask):
    """ Interpolate invalid values pixelwise. Invalid values are defined by a mask of True values.

    Inputs:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). True values indicate invalid values.
    Returns:
        invalid_filled: 3D array of raw image data with invalid values interpolated.
    """
    xs = np.arange(img.shape[0])[~mask]
    invalid_filled = np.copy(img)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            ys = img[:, i, j][~mask]
            interp_f = interpolate.interp1d(
                xs, ys, kind="previous", fill_value="extrapolate")
            missing_xs = np.argwhere(mask).ravel()
            missing_ys = interp_f(missing_xs)
            invalid_filled[mask, i, j] = missing_ys
    return invalid_filled


def plot_image_mean_and_stim(img, mask=None, style="line", duration=0, fs=1):
    """ Plot mean of image (mask optional) and mark points where the image was stimulated

    """
    trace = image_to_trace(img, mask)
    masked_trace, spike_mask = traces.remove_stim_crosstalk(trace)
    stim_end = crosstalk_mask_to_stim_index(spike_mask)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
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
            trace_smoothed = signal.savgol_filter(
                img[:, i, j], savgol_length, 2)
            px_peaks, _ = signal.find_peaks(
                trace_smoothed, prominence=peak_prominence)
            if len(px_peaks) > 1:
                plt.plot(trace_smoothed)
                plt.plot(px_peaks, trace_smoothed[px_peaks], "rx")
                raise ValueError("More than one candidate peak detected")
            elif len(px_peaks) == 1:
                kernel_hits[i, j] = True
                pidx = px_peaks[0]
                if pidx < nbefore:
                    xs = np.arange(nbefore-pidx, kernel_length)
                    ys = img[:pidx-nbefore+kernel_length, i, j]
                    interpf = interpolate.interp1d(
                        xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                elif pidx-nbefore+kernel_length > img.shape[0]:
                    xs = np.arange(img.shape[0] - (pidx-nbefore))
                    ys = img[pidx-nbefore:, i, j]
                    interpf = interpolate.interp1d(
                        xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                else:
                    ktrace = img[pidx-nbefore:pidx-nbefore+kernel_length, i, j]
                ktrace = (ktrace - np.min(ktrace)) / \
                    (np.max(trace_smoothed) - np.min(trace_smoothed))
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

    beta = np.apply_along_axis(lambda tr: kernel_fit_single_trace(tr, kernel, minshift, maxshift, offset_width),
                               0, img)
    error_det = beta[4, :, :]
    beta = beta[:4, :, :]
    failed_pixels = np.sum(np.isnan(error_det))
    print("%d/%d pixels failed to fit (%.2f %%)" %
          (failed_counter, height*width, failed_counter/(height*width)*100))


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
        popt, pcov = optimize.curve_fit(utils.shiftkern, kernel, trace,
                                        p0=[1, 1, 1, np.random.randint(-offset_width, offset_width+1)], absolute_sigma=True,
                                        bounds=([0, -np.inf, 0, minshift], [np.inf, np.inf, np.inf, maxshift]))
        beta = popt
        error_det = np.linalg.det(pcov)
    except Exception as e:
        beta = np.nan*np.ones(4)
        error_det = np.nan
    beta = np.append(beta, error_det)
    return beta


def spline_fit_single_trace(trace, s, knots, plot=False, n_iterations=100, eps=0.01, ax1=None):
    """ Least squares spline fitting of a single timeseries
    """
    x = np.arange(len(trace))
    trace[np.isnan(trace)] = np.min(trace)
    (t, c, k), res, _, _ = interpolate.splrep(
        x, trace, s=s, task=-1, t=knots, full_output=True, k=3)
    spl = interpolate.BSpline(t, c, k)
    spl_values = spl(x)
    if plot:
        if ax1 is None:
            fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(trace)
        ax1.plot(spl(x))

    dspl = spl.derivative()
    d2spl = spl.derivative(nu=2)

    def custom_newton_lsq(x, y, bounds=(-np.inf, np.inf)):
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
        beta = np.nan*np.ones(5)
        return beta, spl
    global_max_index = np.argmax(extrema_values)
    global_max = extrema[global_max_index]
    global_max_val = extrema_values[global_max_index]

    d2 = d2spl(extrema)
    maxima_indices = np.argwhere(d2 < 0).ravel()
    ss = np.std(extrema_values[maxima_indices])
    sm = np.mean(extrema_values[maxima_indices])
    # print(ss)
    # print((extrema_values[global_max_index]-sm)/ss)
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
        ax1.plot([last_min_before_max, global_max], spl(
            [last_min_before_max, global_max]), "rx")
    # min_val = spl(last_min_before_max)

    try:
        min_val = np.nanpercentile(spl_values[:global_max], 10)
        halfmax_magnitude = (global_max_val - min_val)/2 + min_val
#         halfmax = optimize.minimize_scalar(lambda x: (spl(x) - halfmax_magnitude)**2, method='bounded', \
#                                        bounds=[last_min_before_max, global_max]).x
        try:
            # hm = np.argmin((spl_values[last_min_before_max:global_max]-\
            # halfmax_magnitude)**2) + last_min_before_max
            reversed_values = np.flip(spl_values[:global_max])
            moving_away_from_hm = np.diff(
                reversed_values-halfmax_magnitude)**2 > 0
            close_to_hm = (
                reversed_values-halfmax_magnitude)**2 < ((global_max_val - min_val)*5/global_max)**2
            hm = global_max - \
                np.argwhere(close_to_hm[:-1] & moving_away_from_hm).ravel()[0]
            # hm = np.argmin((spl_values[:global_max]-\
            # halfmax_magnitude)**2)
#             halfmax = optimize.minimize_scalar(lambda x: (spl(x) - halfmax_magnitude)**2, method='bounded', \
#                                            bounds=[hm-1, hm+1]).x
            hm0 = hm
            for j in range(n_iterations):
                hm1 = custom_newton_lsq(
                    hm0, halfmax_magnitude, bounds=[hm-3, hm+3])
                if (hm1 - hm0)**2 < eps**2:
                    break
                hm0 = hm1
            halfmax = hm1

            local_max_deriv_idx = np.argwhere(
                np.diff(np.sign(d2spl(x)))).ravel()
            local_max_deriv_idx = local_max_deriv_idx[local_max_deriv_idx < global_max]
            if len(local_max_deriv_idx) > 0:
                max_deriv_idx = local_max_deriv_idx[np.argmin(
                    (local_max_deriv_idx - halfmax)**2)]
                max_deriv_interp = max_deriv_idx + \
                    (0-d2spl(max_deriv_idx)) / \
                    (d2spl(max_deriv_idx+1)-d2spl(max_deriv_idx))
            else:
                max_deriv_interp = np.nan

        except IndexError as e:
            # print(e)
            beta = np.nan*np.ones(5)
            return beta, spl
        if plot:
            ax1.plot(halfmax, spl(halfmax), "gx")
            ax1.plot(max_deriv_interp, spl(max_deriv_interp), "bx")
    except Exception as e:
        # print(e)
        beta = np.nan*np.ones(5)
        return beta, spl
    beta = np.array([global_max, halfmax, global_max_val /
                    min_val, res, max_deriv_interp])
    if plot:
        return beta, spl, ax1
    else:
        return beta, spl


def spline_timing(img, s=0.1, n_knots=4, upsample_rate=1):
    """ Perform spline fitting to functional imaging data do determine wavefront timing
    """
    knots = np.linspace(0, img.shape[0]-1, num=n_knots)[1:-1]
    q = np.apply_along_axis(
        lambda tr: spline_fit_single_trace(tr, s, knots), 0, img)
    beta = np.moveaxis(np.array(list(q[0].ravel())).reshape(
        (img.shape[1], img.shape[2], -1)), 2, 0)
    x = np.arange(img.shape[0]*upsample_rate)/upsample_rate
    smoothed_vid = np.array([spl(x) for spl in q[1].ravel()])
    smoothed_vid = np.moveaxis(smoothed_vid.reshape(
        (img.shape[1], img.shape[2], -1)), 2, 0)
    noise_estimate = np.std(img - smoothed_vid, axis=0)
    beta = np.concatenate([beta, noise_estimate[np.newaxis, :, :]], axis=0)
    return beta, smoothed_vid


def process_isochrones(beta, dt, threshold=None, plot=False, intensity_mask=None, threshold_mode="amplitude", med_filt_size=3, opening_size=3, closing_size=3, amplitude_artifact_cutoff=2.5, dilation_size=0, valid_mask=None):
    """ Clean up spline fitting to better visualize isochrones: get rid of nans and low-amplitude values 
    """

    if threshold_mode == "amplitude":
        amplitude = beta[2, :, :]
    elif threshold_mode == "snr":
        amplitude = (np.abs(beta[2] - 1)/beta[5])**2
    if valid_mask is None:
        amplitude_nanr = np.copy(amplitude)
        amplitude_nanr[beta[2] > amplitude_artifact_cutoff] = np.nan
        minval = np.nanmin(amplitude)
        amplitude_nanr[np.isnan(amplitude_nanr)] = minval
        # thresh = (np.percentile(amplitude_nanr,90)-1)*pct_threshold/100+1
        amplitude_nanr = ndimage.gaussian_filter(amplitude_nanr,
                                                 sigma=3)
        if threshold is None:
            threshold = min(
                max(filters.threshold_triangle(amplitude_nanr), 2), 100)
        if intensity_mask is None:
            intensity_mask = np.ones_like(amplitude, dtype=bool)
        # print(threshold)
        # thresh = filters.threshold_otsu(amplitude_nanr)
        mask = (amplitude_nanr > threshold) & intensity_mask
        if plot:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            visualize.display_roi_overlay(
                amplitude_nanr, mask.astype(int), ax=ax1)
        if opening_size > 0:
            mask = morphology.binary_opening(
                mask, selem=morphology.disk(opening_size))
        if closing_size > 0:
            mask = morphology.binary_closing(
                mask, selem=morphology.disk(closing_size))

        labels = measure.label(mask)
        label_values, counts = np.unique(labels, return_counts=True)
        label_values = label_values[1:]
        counts = counts[1:]
        try:
            valid_mask = labels == label_values[np.argmax(counts)]
            if dilation_size > 0:
                valid_mask = morphology.binary_dilation(
                    valid_mask, selem=morphology.disk(dilation_size))
        except Exception as e:
            valid_mask = np.zeros_like(amplitude, dtype=bool)

        if plot:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            visualize.display_roi_overlay(
                amplitude_nanr, mask.astype(int), ax=ax1)

    hm = beta[1, :, :]
    dv = beta[4, :, :]

    hm_nans_removed = remove_nans(hm, kernel_size=13)
    dv_nans_removed = remove_nans(dv, kernel_size=13)

    hm_smoothed = ndimage.median_filter(hm_nans_removed, size=med_filt_size)
    hm_smoothed = ndimage.gaussian_filter(hm_smoothed, sigma=1)
    hm_smoothed[~valid_mask] = np.nan
    hm_smoothed *= dt*1000

    dv_smoothed = ndimage.median_filter(dv_nans_removed, size=med_filt_size)
    dv_smoothed = ndimage.gaussian_filter(dv_smoothed, sigma=1)
    dv_smoothed[~valid_mask] = np.nan
    dv_smoothed *= dt*1000

    return hm_smoothed, dv_smoothed


def clamp_intensity(img, pctiles=[2, 99]):
    min_val, max_val = np.nanpercentile(img, pctiles)
    processed_img = np.copy(img)
    processed_img[processed_img > max_val] = max_val
    processed_img[processed_img < min_val] = min_val
    return processed_img


def normalize_and_clamp(img, pctile=99):
    processed_img = img/np.nanpercentile(img, pctile)
    processed_img[processed_img > 1] = 1
    processed_img[processed_img < 0] = 0
    return processed_img


def remove_nans(img, kernel_size=3):
    """ Replace NaNs in a 2D image with an average of surrounding values
    """
    convinput = np.copy(img)
    convinput[np.isnan(img)] = 0
    kernel = np.ones((kernel_size, kernel_size))/(kernel_size**2-1)
    kernel[kernel_size//2, kernel_size//2] = 0
    nans_removed = np.copy(img)
    nans_removed[np.isnan(img)] = ndimage.convolve(
        convinput, kernel)[np.isnan(img)]
    return nans_removed


def estimate_local_velocity(activation_times, deltax=7, deltay=7, deltat=100, valid_points_threshold=10, debug=False, weights=None):
    """ Use local polynomial fitting strategy from Bayley et al. 1998 to determine velocity from activation map
    """
    X, Y = np.meshgrid(np.arange(activation_times.shape[0]), np.arange(
        activation_times.shape[1]))
    coords = np.array([Y.ravel(), X.ravel()]).T
    Tsmoothed = np.ones_like(activation_times)*np.nan
    residuals = np.ones_like(activation_times)*np.nan
    # Note v = (v_y, v_x), consistent with row and column convention
    v = np.ones(
        (2, activation_times.shape[0], activation_times.shape[1]))*np.nan
    n_points_fit = np.zeros_like(activation_times)

    for y, x in coords:
        # if np.isnan(activation_times[y, x]):
        #     continue
        local_X, local_Y = np.meshgrid(np.arange(max(0, x-deltax), min(activation_times.shape[1], x+deltax+1)),
                                       np.arange(max(0, y-deltay), min(activation_times.shape[0], y+deltay+1)))\
            - np.array([x, y])[:, np.newaxis, np.newaxis]
        local_times = activation_times[max(0, y-deltay):min(activation_times.shape[0], y+deltay+1),
                                       max(0, x-deltax):min(activation_times.shape[1], x+deltax+1)]
        local_X = local_X.ravel()
        local_Y = local_Y.ravel()
        A = np.array([local_X**2, local_Y**2, local_X*local_Y,
                     local_X, local_Y, np.ones_like(local_X)]).T
        if weights is None:
            w = np.ones((A.shape[0], 1))
        else:
            w = weights[max(0, y-deltay):min(activation_times.shape[0], y+deltay+1),
                        max(0, x-deltax):min(activation_times.shape[1], x+deltax+1)].ravel()[:, None]
        b = local_times.ravel()
        w = w[~np.isnan(b), :]
        A = A[~np.isnan(b), :]
        b = b[~np.isnan(b)]
        mask = np.abs(b - activation_times[y, x]) < deltat
        # print(w.shape, A.shape, b.shape)
        w = w[mask, :]
        A = A[mask, :] * np.sqrt(w)
        b = b[mask] * np.sqrt(w).ravel()
        # print(w.shape, A.shape, b.shape)
        n_points_fit[y, x] = len(b)
        if n_points_fit[y, x] < valid_points_threshold:
            continue
        try:
            p, res, _, _ = np.linalg.lstsq(A, b)
            Tsmoothed[y, x] = p[5]
            residuals[y, x] = res
            # To make v = (v_y, v_x, flip)
            v[:, y, x] = np.flip(p[3:5]/(p[3]**2 + p[4]**2))
        except Exception as e:
            pass
    if debug:
        return v, Tsmoothed, n_points_fit
    else:
        return v, Tsmoothed


def correct_photobleach(img, mask=None, method="localmin", nsamps=51, amplitude_window=0.5, dt=0.01, invert=False, return_params=False):
    """ Perform photobleach correction on each pixel in an image
    """
    if method == "linear":
        # Perform linear fit to each pixel independently over time
        corrected_img = np.zeros_like(img)
        for y in range(img.shape[1]):
            for x in range(img.shape[2]):
                corrected_trace, _ = traces.correct_photobleach(img[:, y, x])
                corrected_img[:, y, x] = corrected_trace

    elif method == "localmin":
        #
        if invert:
            mean_img = img.mean(axis=0)
            raw_img = 2*mean_img - img
        else:
            raw_img = img
        mean_trace = image_to_trace(raw_img, mask)
        _, pbleach = traces.correct_photobleach(
            mean_trace, method=method, nsamps=nsamps)
        corrected_img = np.divide(raw_img, pbleach[:, np.newaxis, np.newaxis])
        if return_params:
            return corrected_img, pbleach
        # print(corrected_img.shape)

    elif method == "monoexp":
        mean_trace = image_to_trace(img, mask)
        background_level = np.percentile(img, 5, axis=0)
        _, pbleach, params = traces.correct_photobleach(mean_trace,
                                                        method="monoexp", return_params=True, invert=invert, a=0.1, b=5e-4)
        dur = img.shape[0]
        amplitude_window_idx = int(amplitude_window/dt)
        max_val = np.median(img[:amplitude_window_idx], axis=0)
        min_val = np.median(img[-amplitude_window_idx:], axis=0)
        amplitude = (max_val - min_val)/(1-np.exp(params[1]*dur))
        amplitude[~np.isfinite(amplitude)] = 0
        amplitude = np.maximum(amplitude, 0)
        amplitude = filters.median(amplitude, selem=morphology.disk(7))
        amplitude[amplitude/params[0]*pbleach[-1] > background_level] = 0
        # pctile = np.percentile(amplitude, 99)
        # amplitude[amplitude > pctile] = pctile

        corrected_img = img - np.divide(amplitude, params[0], out=np.zeros_like(amplitude), where=params[0] > 5e-2)\
            * pbleach[:, np.newaxis, np.newaxis]
        # if invert:
        #     corrected_img = 2*np.mean(corrected_img, axis=0) - corrected_img

        if return_params:
            return corrected_img, amplitude, params
    elif method == "biexp":

        mean_trace = image_to_trace(img, mask)
        background_level = np.percentile(img, 5)

        _, pbleach, params = traces.correct_photobleach(mean_trace,
                                                        method="biexp", return_params=True, invert=invert)
        dur = img.shape[0]
        amplitude_window_idx = int(amplitude_window/dt)

        max_val = np.median(img[:amplitude_window_idx], axis=0)
        min_val = np.median(img[-amplitude_window_idx:], axis=0)
        amplitude = (max_val - min_val)/(1-np.exp(params[1]*dur))
        amplitude[~np.isfinite(amplitude)] = 0
        amplitude = np.maximum(amplitude, 0)
        amplitude = filters.median(amplitude, selem=morphology.disk(7))
        pctile = np.percentile(amplitude, 99)
        amplitude[amplitude > pctile] = pctile

        corrected_img = img - np.divide(amplitude, params[0], out=np.zeros_like(amplitude), where=params[0] > 5e-2)\
            * pbleach[:, np.newaxis, np.newaxis]

        # if invert:
        #     corrected_img = 2*np.mean(corrected_img, axis=0) - corrected_img
        if return_params:
            return corrected_img, amplitude, params
    elif method == "decorrelate":
        # Zero the mean over time
        # mean_img = img.mean(axis=0)
        mean_trace = img.mean(axis=(1, 2))
        corrected_img = regress_video(img, mean_trace)

    else:
        raise ValueError("Not Implemented")

    return corrected_img


def get_image_dFF(img, baseline_percentile=10, t_range=(0, -1)):
    """ Convert a raw image into dF/F

    """
    baseline = np.percentile(
        img[t_range[0]:t_range[1]], baseline_percentile, axis=0)
    dFF = img/baseline
    return dFF


def get_spike_videos(img, peak_indices, before, after, normalize_height=True):
    """ Generate spike-triggered videos of a defined length from a long video and peak indices

    """
    spike_imgs = np.ones((len(peak_indices), before+after,
                         img.shape[1], img.shape[2]))*np.nan
    for pk_idx, pk in enumerate(peak_indices):
        before_pad_length = max(before - pk, 0)
        after_pad_length = max(0, pk+after - img.shape[0])
        spike_img = np.concatenate([np.ones((before_pad_length, img.shape[1], img.shape[2]))*np.nan,
                                    img[max(0, pk-before):min(img.shape[0], pk+after), :, :],
                                    np.ones((after_pad_length, img.shape[1], img.shape[2]))*np.nan])
        if normalize_height:
            spike_img /= np.nanpercentile(spike_img, 99)
        try:
            spike_imgs[pk_idx, :, :, :] = spike_img
        except Exception as e:
            print(spike_imgs.shape)
            print(pk)
            print(before_pad_length, after_pad_length)
            raise e
    return spike_imgs


def spike_triggered_average_video(img, peak_indices, before, after, include_mask=None, normalize_height=False, full_output=False):
    """ Create a spike-triggered average video
    """
    if include_mask is None:
        include_mask = np.ones_like(peak_indices, dtype=bool)
    spike_triggered_images = get_spike_videos(
        img, peak_indices[include_mask], before, after, normalize_height=normalize_height)
    sta = np.nanmean(spike_triggered_images, axis=0)
    if full_output:
        return sta, spike_triggered_images
    else:
        return sta, None


def test_isochron_detection(vid, x, y, savgol_window=25, figsize=(8, 3)):
    """ Check detection of half-maximum in spike_triggered averages
    """

    trace = vid[:, y, x]
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
        smoothed_vid = np.apply_along_axis(
            lambda x: signal.savgol_filter(x, savgol_window, 2), 0, vid)
    else:
        smoothed_vid = vid

    zeroed = smoothed_vid - smoothed_vid[0, :, :]
    normed = zeroed/zeroed.max(axis=0)
    hm_indices = np.apply_along_axis(find_half_max, 0, normed)
    chron = hm_indices*dt*1000
    return chron


def analyze_wave_dynamics(beta, dt, um_per_px, mask_function_tsmoothed=None,
                          deltax=9, deltat=350, **isochrone_process_params):
    """ Measure spatiotemporal properties of wave propagation 
    """

    def default_mask(Ts, divergence):
        nan_mask = morphology.binary_dilation(np.pad(np.isnan(Ts), 1, constant_values=True),
                                              selem=morphology.disk(3))

        return np.ma.masked_array(Ts, nan_mask[1:-1, 1:-1] |
                                  (divergence < 0.5))
    if mask_function_tsmoothed is None:
        mask_function_tsmoothed = default_mask
    hm_nan, dv_max_nan = process_isochrones(beta, dt,
                                            threshold_mode="snr", **isochrone_process_params)
    if np.sum(~np.isnan(hm_nan)) == 0:
        return None
    snr = np.nan_to_num((np.abs(beta[2]-1)/beta[5])**2)
    _, Tsmoothed = estimate_local_velocity(
        hm_nan, deltax=deltax, deltay=deltax, deltat=deltat, weights=snr)
    v, Tsmoothed_dv = estimate_local_velocity(
        dv_max_nan, deltax=deltax, deltay=deltax, deltat=deltat, weights=snr)
    v *= um_per_px*1000
    abs_vel = np.linalg.norm(v, axis=0)
    mean_velocity = np.nanmean(abs_vel.ravel())
    median_velocity = np.nanmedian(abs_vel.ravel())
    divergence = utils.div(v/abs_vel)

    masked_tsmoothed = mask_function_tsmoothed(Tsmoothed, divergence)
    masked_tsmoothed_dv = mask_function_tsmoothed(Tsmoothed_dv, divergence)

    min_activation_time = np.unravel_index(
        np.ma.argmin(masked_tsmoothed), Tsmoothed.shape)
    min_activation_time_dv = np.unravel_index(
        np.ma.argmin(masked_tsmoothed_dv), Tsmoothed_dv.shape)

    results = (mean_velocity, median_velocity, min_activation_time[1],
               min_activation_time[0], min_activation_time_dv[1], min_activation_time_dv[0])
    return results, Tsmoothed, Tsmoothed_dv, divergence, v


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
            smoothed = np.apply_over_axes(lambda a, axis: np.apply_along_axis(
                lambda x: signal.sosfiltfilt(sos, x), axis, a), raw_img, [1, 2])
            di = smoothed[:, np.arange(
                smoothed.shape[1], step=downsample_factor, dtype=int), :]
            di = di[:, :, np.arange(
                smoothed.shape[2], step=downsample_factor, dtype=int)]
        elif aa == "gaussian":
            smoothed = ndimage.gaussian_filter(
                raw_img, [0, downsample_factor, downsample_factor])
            di = smoothed[:, np.arange(
                smoothed.shape[1], step=downsample_factor, dtype=int), :]
            di = di[:, :, np.arange(
                smoothed.shape[2], step=downsample_factor, dtype=int)]
        else:
            di = transform.downscale_local_mean(
                raw_img, (1, downsample_factor, downsample_factor))
    return di


def get_image_dff_corrected(img, nsamps_corr, mask=None, plot=None, full_output=False):
    """ Convert image to Delta F/F after doing photobleach correction and sampling 
    """

    # Generate mask if not provided
    mean_img = img.mean(axis=0)
    if mask is None:
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0]/50)
        mask = morphology.binary_closing(
            mask, selem=np.ones((kernel_size, kernel_size)))

    if plot is not None:
        _, _ = visualize.display_roi_overlay(
            mean_img, mask.astype(int), ax=plot)
    # Correct for photobleaching and convert to DF/F
    pb_corrected_img = correct_photobleach(img, mask=np.tile(
        mask, (img.shape[0], 1, 1)), nsamps=nsamps_corr)
    dFF_img = get_image_dFF(pb_corrected_img)

    if full_output:
        return dFF_img, mask
    else:
        return dFF_img


def image_to_peaks(img, mask=None, prom_pct=90, exclude_pks=None, fs=1, min_width_s=0.5, f_c=2.5, **peak_detect_params):
    """ Detect spike events in image
    """

    if mask is None:
        mean_img = img.mean(axis=0)
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0]/50)
        mask = morphology.binary_closing(
            mask, selem=np.ones((kernel_size, kernel_size)))

    sos = signal.butter(5, f_c, btype="lowpass",
                        output="sos", fs=fs)

    mask_trace = image_to_trace(img, mask=np.tile(mask, (img.shape[0], 1, 1)))
    mask_trace = signal.sosfiltfilt(sos, mask_trace)
    ps = np.percentile(mask_trace, [10, prom_pct])
    prominence = ps[1] - ps[0]

    pks, _ = signal.find_peaks(mask_trace, prominence=prominence, width=(
        int(min_width_s*fs), None), rel_height=0.8, **peak_detect_params)

    if exclude_pks:
        keep = np.ones(len(pks), dtype=bool)
        keep[exclude_pks] = 0
        pks = pks[keep]

    return pks, mask_trace


def image_to_sta(raw_img, fs=1, mask=None, plot=False,
                 savedir=None, prom_pct=90, sta_bounds="auto",
                 exclude_pks=None, offset=0, normalize_height=True,
                 full_output=False, min_width_s=0.5):
    """ Convert raw image into spike triggered average
    """
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    if plot:
        fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()

    # Generate mask if not provided
    mean_img = raw_img.mean(axis=0)
    if mask is None:
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0]/50)
        mask = morphology.binary_closing(
            mask, selem=np.ones((kernel_size, kernel_size)))
    _ = visualize.display_roi_overlay(mean_img, mask.astype(int), ax=axes[0])

    raw_trace = image_to_trace(
        raw_img, np.tile(mask, (raw_img.shape[0], 1, 1)))

    # convert to DF/F

    dFF_img = get_image_dFF(raw_img)
    pks, dFF_mean = image_to_peaks(
        dFF_img-1, mask=mask, prom_pct=prom_pct, fs=fs, exclude_pks=exclude_pks, min_width_s=min_width_s)

    if savedir:
        skio.imsave(os.path.join(savedir, "dFF.tif"),
                    dFF_img.astype(np.float32))
        skio.imsave(os.path.join(savedir, "dFF_display.tif"), exposure.rescale_intensity(
            dFF_img, out_range=(0, 255)).astype(np.uint8))

    dFF_mean = image_to_trace(
        dFF_img, mask=np.tile(mask, (dFF_img.shape[0], 1, 1)))
    axes[1].plot(np.arange(dFF_img.shape[0])/fs, dFF_mean, color="C2")

    if plot:
        axes[1].plot(np.arange(dFF_img.shape[0])/fs, dFF_mean)
        axes[1].plot(pks/fs, dFF_mean[pks], "rx")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel(r"$F/F_0$")
        tx = axes[1].twinx()
        tx.plot(np.arange(dFF_img.shape[0])/fs, raw_trace, color="C1")
        tx.set_ylabel("Mean counts")

    # Automatically determine bounds for spike-triggered average
    if sta_bounds == "auto":
        try:
            # Align all detected peaks of the full trace and take the mean
            aligned_traces = traces.align_fixed_offset(
                np.tile(dFF_mean, (len(pks), 1)), pks)
        except Exception as e:
            print(e)
            return raw_trace
        mean_trace = np.nanmean(aligned_traces, axis=0)
        # Smooth
        spl = interpolate.UnivariateSpline(np.arange(len(mean_trace)),
                                           np.nan_to_num(mean_trace, nan=np.nanmin(mean_trace)), s=0.001)
        smoothed = spl(np.arange(len(mean_trace)))
        smoothed_dfdt = spl.derivative()(np.arange(len(mean_trace)))
        # Find the first minimum before and the first minimum after the real spike-triggered peak
        if plot:
            axes[3].plot(mean_trace)
            axes[3].plot(smoothed, color="C1")
            tx = axes[3].twinx()
            tx.plot(smoothed_dfdt, color="C2")
            tx.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        try:
            minima_left = np.argwhere((smoothed_dfdt < 0)).ravel()
            minima_right = np.argwhere((smoothed_dfdt > 0)).ravel()
            # calculate number of samples to take before and after the peak
            b1 = pks[-1] - minima_left[minima_left < pks[-1]][-1]+offset
            b2 = minima_right[minima_right > pks[-1]][0] - pks[-1]+offset
        except Exception:
            return mean_trace

        if plot:
            axes[3].plot([pks[-1]-b1, pks[-1], pks[-1]+b2],
                         mean_trace[[pks[-1]-b1, pks[-1], pks[-1]+b2]], "rx")
    else:
        b1, b2 = sta_bounds

    # Collect spike traces according to bounds
    sta_trace = traces.get_sta(
        dFF_mean, pks, b1, b2, f_s=fs, normalize_height=normalize_height)

    # Get STA video

    sta, spike_images = spike_triggered_average_video(dFF_img, pks, b1,
                                                      b2, normalize_height=normalize_height, full_output=full_output)

    if plot:
        axes[2].plot(np.arange(b1+b2)/fs, sta_trace)
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel(r"$F/F_0$")
        axes[2].set_title("STA taps: %d + %d" % (b1, b2))
        plt.tight_layout()
    if savedir:
        skio.imsave(os.path.join(savedir, "sta.tif"), sta)
        if plot:
            plt.savefig(os.path.join(savedir, "QA_plots.tif"))
        with open(os.path.join(savedir, "temporal_statistics.pickle"), "wb") as f:
            temp_stats = {}
            temp_stats["freq"] = len(pks)/dFF_img.shape[0]*fs
            isi = np.diff(pks)/fs
            temp_stats["isi_mean"] = np.nanmean(isi)
            temp_stats["isi_std"] = np.nanstd(isi)
            temp_stats["n_pks"] = len(pks)
            pickle.dump(temp_stats, f)
    return sta, spike_images

# SEGMENTATION


def identify_hearts(img, prev_coms=None, prev_mask_labels=None, fill_missing=True,
                    band_bounds=(0.1, 2), opening_size=5,
                    dilation_size=15, method="freq_band", corr_threshold=0.9, bbox_offset=5,
                    **initial_segmentation_args):
    """ Pick out hearts from widefield experiments using PCA and power content in frequency band
    """
#     print("band_bounds:", band_bounds)
#     print("band_threshold:", band_threshold)
#     print("opening_size:", opening_size)
#     print("dilation_size:", dilation_size)
#     print("f_s:", f_s)
    mean_img = img.mean(axis=0)
    zeroed_image = img-mean_img

    if method == "freq_band":
        initial_guesses = segment_by_frequency_band(
            img, band_bounds, **initial_segmentation_args)
    elif method == "pca_moments":
        initial_guesses = segment_by_pca_moments(
            img, **initial_segmentation_args)
    elif callable(method):
        initial_guesses = method(img)
    else:
        raise Exception("Invalid segmentation method")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    visualize.display_roi_overlay(mean_img, initial_guesses, ax=ax1)
    bboxes = [p["bbox"] for p in measure.regionprops(initial_guesses)]
    new_mask = np.zeros_like(mean_img, dtype=bool)
#     new_mask_labels = np.deepcopy(initial_guesses)
    n_rois = np.max(initial_guesses)
    corr_img = np.zeros(
        (n_rois, mean_img.shape[0], mean_img.shape[1]), dtype=float)
    if corr_threshold == "None":
        new_mask = initial_guesses.astype(bool)
    else:
        corr_threshold = float(corr_threshold)
        for i in range(1, n_rois+1):
            bbox = bboxes[i-1]
            r1 = max(bbox[0]-bbox_offset, 0)
            c1 = max(bbox[1]-bbox_offset, 0)
            r2 = min(bbox[2]+bbox_offset, initial_guesses.shape[0])
            c2 = min(bbox[3]+bbox_offset, initial_guesses.shape[1])
    #         print(r1,r2,c1,c2)
            roi_mask = np.zeros_like(initial_guesses, dtype=bool)
            roi_mask[r1:r2, c1:c2] = 1
            initial_trace = image_to_trace(zeroed_image, mask=roi_mask)
            roi_traces = zeroed_image[:, roi_mask]
    #         print(roi_traces.shape)
            corrs = np.apply_along_axis(lambda x: stats.pearsonr(
                initial_trace, x)[0], 0, roi_traces)
            corrs = corrs.reshape((r2-r1, c2-c1))
            corr_img[i-1, r1:r2, c1:c2] = corrs

        corr_mask = morphology.binary_opening(np.max(
            corr_img > corr_threshold, axis=0), selem=np.ones((opening_size, opening_size)))
        new_mask = morphology.binary_dilation(
            corr_mask, selem=morphology.disk(dilation_size))
    new_mask_labels = measure.label(new_mask)
    coms = ndimage.center_of_mass(
        new_mask, labels=new_mask_labels, index=np.arange(1, np.max(new_mask_labels)+1))
    coms = np.array(coms)

    # print(coms.shape)
    return new_mask_labels, coms


def segment_by_frequency_band(img, band_bounds, f_s=1, band_threshold=0.45,
                              block_size=375, offset=5, manual_intensity_mask=None):
    """ Get segmentation by relative power content within a range of frequencies
    """
    mean_img = img.mean(axis=0)
    zeroed_image = img-mean_img
    if manual_intensity_mask is None:
        local_thresh = filters.threshold_local(
            mean_img, block_size=block_size, offset=offset)
        # intensity_mask = mean_img > np.percentile(mean_img, intensity_threshold*100)
        intensity_mask = mean_img > local_thresh
    else:
        intensity_mask = manual_intensity_mask

    pixelwise_fft = fft(zeroed_image, axis=0)
    N_samps = img.shape[0]
    fft_freq = fftfreq(N_samps, 1/f_s)[:N_samps//2]
    abs_power = np.abs(pixelwise_fft[:N_samps//2, :, :])**2
    norm_abs_power = abs_power/np.sum(abs_power, axis=0)
    band_power = np.sum(norm_abs_power[(fft_freq > band_bounds[0]) & (
        fft_freq < band_bounds[1]), :, :], axis=0)
    smoothed_band_power = filters.median(
        band_power, selem=np.ones((5, 5)))*intensity_mask.astype(int)
    processed_band_power = morphology.binary_opening(
        (smoothed_band_power > band_threshold), selem=np.ones((3, 3)))
    return measure.label(processed_band_power)


def segment_by_pca_moments(img, n_components=40, krt_threshold=40,
                           skw_threshold=0, ev_threshold=0.3, comp_mag_pct=95):
    """ Segment a widefield video with multiple embryos using moments of principal components (heart flickers will be small compared to FOV)
    """
    mean_img = img.mean(axis=0)
    zeroed_img = img-mean_img
    rd = zeroed_img.reshape((zeroed_img.shape[0], -1))
    # print(rd.shape)
    n_components = min(n_components, rd.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(rd)
    comp_magnitudes = np.abs(pca.components_)
    valid_components = (stats.kurtosis(comp_magnitudes, axis=1) > krt_threshold) &\
        (stats.skew(comp_magnitudes, axis=1) > skw_threshold) &\
        (pca.explained_variance_ratio_ > ev_threshold)
    mask = np.zeros_like(mean_img)
    for comp_idx in np.argwhere(valid_components).ravel():
        comp = comp_magnitudes[comp_idx].reshape(img.shape[1], img.shape[2])
        mask += comp > np.percentile(comp, comp_mag_pct)
    mask = (mask > 0)
    return measure.label(mask)


def get_heart_mask(img: npt.NDArray, n_components: int = 10,
                   krt_thresh: float = 40,
                   corr_thresh: float = 0.8, plot=False):
    """ Extract a binary mask identifying a heart from an image of a
    single embryo.

    Args:
        img: A 3D array of shape (n_frames, n_rows, n_cols) containing a video
            of a single embryo.
        n_components: Number of principal components to use for segmentation.
        krt_thresh: Kurtosis threshold for identifying principal components
            corresponding to heart dynamics.
        corr_thresh: Correlation threshold for identifying pixels corresponding
            to the principal components.
        plot: If True, plot the results of the segmentation.
    Returns:
        A 2D boolean array of shape (n_rows, n_cols) containing the mask.
    """
    pca = PCA(n_components=n_components)
    datmatrix = np.copy(img).reshape(img.shape[0], -1)
    datmatrix -= np.mean(datmatrix, axis=0)
    pca.fit(datmatrix)
    krt = stats.kurtosis(pca.components_, axis=1)
    if plot:
        plot_pca_data(pca, datmatrix, img.shape[1:], n_components=n_components,
                      pc_title= lambda i, comp: f"kurtosis: {krt[i]:.2f}")
    n_valid_components = np.sum(krt > krt_thresh)

    if n_valid_components == 0:
        return np.zeros((img.shape[1], img.shape[2]), dtype=bool)
    else:
        comp_idx = np.argmax(krt)
        comp = np.abs(pca.components_[comp_idx].reshape(
            img.shape[1], img.shape[2]))
        rough_mask = comp > np.percentile(comp, 95)
        test_trace = image_to_trace(img, mask=rough_mask)
        corrs = np.apply_along_axis(
            lambda x: stats.pearsonr(test_trace, x)[0], 0, datmatrix)
        corrs = corrs.reshape(img.shape[1], img.shape[2])
        mask = corrs > corr_thresh
        return mask


def refine_segmentation_pca(img, rois, n_components=10, threshold_percentile=70):
    """ Refine manual segmentation to localize the heart using PCA assuming that transients are the largest fluctuations present.
    Returns cropped region images and masks for each unique region in the global image mask.

    """
    def pca_component_select(pca):
        """ TBD: a smart way to decide the number of principal components. for now just return 1.
        """
        n_components = 1
        selected_components = np.zeros_like(pca.components_[0])
        fraction_explained_variance = pca.explained_variance_ / \
            np.sum(pca.explained_variance_)
        for comp_index in range(n_components):
            selected_components = selected_components + \
                pca.components_[comp_index] * \
                fraction_explained_variance[comp_index]
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

            indices_to_keep = selected_components > np.percentile(
                selected_components, threshold_percentile)
    #         print(gc.shape)
    #         print(indices_to_keep.shape)
            mask = np.zeros((img.shape[1], img.shape[2]), dtype=bool)
            mask[gc[:, 0], gc[:, 1]] = indices_to_keep
        except ValueError as e:
            roi_indices = np.unique(rois)
            roi_indices = roi_indices[roi_indices != 0]
            mask = rois == roi_indices[region_idx]

        selem = np.ones((3, 3), dtype=bool)
        mask = morphology.binary_opening(mask, selem)
        region_masks.append(mask)

    return region_masks


def segment_widefield_series(filepaths, **segmentation_args):
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
    for idx, f in enumerate(filepaths):
        try:
            del raw
        except Exception as e:
            pass
        # print(f)
        try:
            raw = skio.imread(f)
            if img_shape is None:
                img_shape = raw.shape[1:3]
            else:
                if raw.shape[1:3] != img_shape:
                    raise Exception("Incorrect image shape")
        except Exception as e:
            # print(e)
            # exclude_from_write[idx] = 1
            raise e
            continue
        try:
            curr_labels, curr_coms = identify_hearts(raw,
                                                     **segmentation_args)
        except Exception as e:
            # raise e
            print(e)
            print(curr_coms)
            print(f)
            # raise e
            pass
        if curr_labels is None:
            frames.append(np.zeros((raw.shape[1], raw.shape[2]), dtype=int))
        else:
            frames.append(np.copy(curr_labels))

    vid = np.array(frames)
    return vid, exclude_from_write


def link_frames(curr_labels, prev_labels, prev_coms, radius=15, propagate_old_labels=True):
    """ Connect two multi-ROI segmentations adjacent in time
    """
    curr_mask = curr_labels > 0
    all_curr_labels = np.unique(curr_labels)[1:]
    all_prev_labels = np.unique(prev_labels)[1:]

    curr_coms = ndimage.center_of_mass(
        curr_mask, labels=curr_labels, index=all_curr_labels)
    curr_coms = np.array(curr_coms)
    if len(curr_coms.shape) == 2 and len(prev_coms) > 0:
        curr_coms = curr_coms[:, 0] + 1j*curr_coms[:, 1]

        mindist, mindist_indices = utils.pairwise_mindist(curr_coms, prev_coms)
        link_curr = np.argwhere(mindist < radius).ravel()

        link_prev = set(mindist_indices[link_curr]+1)

        link_curr = set(link_curr+1)
    elif len(curr_coms.shape) == 1 or len(prev_coms) == 0:
        link_curr = set([])
        link_prev = set([])
    all_curr_labels_set = set(all_curr_labels)
    all_prev_labels_set = set(all_prev_labels)

    new_labels = np.zeros_like(curr_labels)

    for label in link_curr:
        new_labels[curr_labels ==
                   label] = all_prev_labels[mindist_indices[label-1]]

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
        if np.all(new_labels[curr_labels == label] == 0):
            new_labels[curr_labels == label] = starting_idx + new_rois_counter
            new_rois_counter += 1
    new_mask = new_labels > 0
    new_coms = ndimage.center_of_mass(
        new_mask, labels=new_labels, index=np.unique(new_labels)[1:])
    new_coms = np.array(new_coms)
    try:
        new_coms = new_coms[:, 0] + 1j*new_coms[:, 1]
    except Exception as e:
        new_coms = []
    return new_labels, new_coms


def link_stack(stack, step=-1, radius=15, propagate_old_labels=True):
    if step < 0:
        curr_t = 1
    else:
        curr_t = stack.shape[0]-2

    prev_labels = stack[curr_t+step]
    prev_mask = prev_labels > 0

    prev_coms = ndimage.center_of_mass(
        prev_mask, labels=prev_labels, index=np.arange(1, np.max(prev_labels)+1))
    prev_coms = np.array(prev_coms)
    if len(prev_coms.shape) == 2:
        prev_coms = prev_coms[:, 0] + 1j*prev_coms[:, 1]
    new_labels = [prev_labels]
    while curr_t >= 0 and curr_t < stack.shape[0]:
        curr_labels = stack[curr_t]
        try:
            curr_labels, curr_coms = link_frames(curr_labels, prev_labels, prev_coms,
                                                 radius=radius, propagate_old_labels=propagate_old_labels)
        except Exception as e:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(curr_labels)
            print(prev_coms)
            raise e
        prev_labels = curr_labels
        prev_coms = curr_coms
        curr_t -= step
        new_labels.append(curr_labels)
    new_labels = np.array(new_labels)
    return new_labels


def filter_by_appearances(linked_vid, unlinked_vid, threshold=1/3):
    roi_found = []
    for roi in np.arange(1, np.max(linked_vid)+1):
        roi_linked = linked_vid == roi
        found_in_unlinked = []
        for i in range(roi_linked.shape[0]):
            detected = unlinked_vid[i][roi_linked[i]]
            found_in_unlinked.append(np.any(detected > 0)*(len(detected) > 0))
        roi_found.append(found_in_unlinked)
    roi_found = np.array(roi_found)
    keep = np.argwhere(np.sum(roi_found, axis=1) >
                       threshold*linked_vid.shape[0]).ravel()+1

    filtered_vid = np.zeros_like(linked_vid)
    for idx, roi in enumerate(keep):
        filtered_vid[linked_vid == roi] = idx+1

    return filtered_vid


def fill_missing(vid, threshold=100):
    """ Detect when an ROI drops out of a sequence of movies
    """
    roi_sizes = []
    for roi in range(1, np.max(vid)+1):
        roi_sizes.append(np.sum(vid == roi, axis=(1, 2)))
    roi_sizes = np.array(roi_sizes).reshape(-1, vid.shape[0])

    below_threshold = roi_sizes < threshold
    try:
        closest_above_threshold = np.apply_along_axis(
            utils.closest_non_zero, 1, ~below_threshold).squeeze()
    except Exception as e:
        return vid
    closest_above_threshold = closest_above_threshold.reshape(np.max(vid), -1)
    filled_vid = np.zeros_like(vid)
    for i in range(1, closest_above_threshold.shape[0]+1):
        replaced_vals = vid[closest_above_threshold[i-1], :, :] == i
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
        rr, cc = draw.disk(tuple(centroids[i, :]), radius)
        valid_indices = (rr < refined_mask.shape[0]) & (
            cc < refined_mask.shape[1]) & (rr >= 0) & (cc >= 0)
        rr = rr[valid_indices]
        cc = cc[valid_indices]
        area_regularized_mask[rr, cc] = mask_labels[i]

    return area_regularized_mask


def watershed_mask(mask):
    """
    Perform watershedding on a mask
    """
    dist = ndimage.distance_transform_edt(mask)
    guess_objs = measure.label(mask)
    guess_props = pd.DataFrame(measure.regionprops_table(guess_objs,
                                                         properties=('label', 'equivalent_diameter_area')))
    median_diameter = np.median(guess_props['equivalent_diameter_area'])
    # print(median_diameter)
    coords = feature.peak_local_max(dist, footprint=morphology.disk(
        median_diameter/2*0.8), labels=mask, min_distance=int(median_diameter*0.5), exclude_border=5)
    # fig1, ax1 = plt.subplots(figsize=(4,4))
    # ax1.imshow(dist)
    # ax1.plot(coords[:,1], coords[:,0], "wx")
    # print(coords)
    seeds = np.zeros(dist.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    seeds = measure.label(seeds)

    labels = segmentation.watershed(-dist, seeds, mask=mask)
    return labels


def segment_whole_embryos(img, block_size, offset, min_obj_size, **threshold_params):
    """
    Segment whole embryos (not hearts only) from an array experiment
    """
    threshold = filters.threshold_local(
        img, block_size, offset=offset, **threshold_params)
    mask = img > threshold
    radius = 7
    mask = utils.pad_func_unpad(mask, lambda x: morphology.binary_opening(
        x, footprint=morphology.disk(radius)), radius, constant_values=0)
    mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
    # Watershed
    regions = watershed_mask(mask)
    return mask, regions


def calculate_dish_rotation(centroids):
    """
    Calculate angle of rotation of a dish to make all embryos face upwards
    """
    _, mindist_indices = utils.pairwise_mindist(centroids, centroids)
    direction_vectors = centroids - centroids[mindist_indices]
#     print(direction_vectors.shape)
    # Enforce x positive
    sgn = np.ones((direction_vectors.shape[0])) - \
        2*(direction_vectors[:, 0] < 0)
    direction_vectors *= sgn[:, np.newaxis]
#     print(direction_vectors)
    theta = - \
        np.median(
            np.arctan2(-direction_vectors[:, 1], direction_vectors[:, 0]))
    return theta


def remap_regions(regions, props):
    """
    Renumber regions based on known array formation (centroids may be slightly different but we want to fix them). Column changing fastest.
    """
    centroid_y = props["centroid-0"].to_numpy()
    # print(centroid_y)
    cl = cluster.AffinityPropagation().fit(centroid_y.reshape(-1, 1))
    regularized_centroid_y = np.zeros(props.shape[0])
    for label in range(np.max(cl.labels_)+1):
        regularized_centroid_y[cl.labels_ == label] = np.mean(
            centroid_y[cl.labels_ == label])
    props["regularized_centroid-0"] = regularized_centroid_y
    props = props.sort_values(by=["regularized_centroid-0", "centroid-1"])
    remapped_regions = map_array(
        regions, props["label"].to_numpy(), np.arange(props.shape[0])+1)
    return remapped_regions


def split_embryos(img, block_size=201, offset=0, extra_split_arrays=[],
                  min_obj_size=1000, manual_roi_seeds=None, manual_bbox_size=None, **threshold_params):
    """ Use assumption that embryos lie on a grid to split a widefield video into videos for individual embryos. This is useful when panning between multiple FOVs that may not necessarily be aligned.
    """
    mean_img = img.mean(axis=0)
    mask, regions = segment_whole_embryos(
        mean_img, block_size, offset, min_obj_size)
    # fig1, ax1 = plt.subplots(figsize=(4,4))
    # ax1.imshow(regions)
    if manual_roi_seeds is not None:
        for r in range(1, regions.max()+1):
            if np.sum(manual_roi_seeds[regions == r]) == 0:
                regions[regions == r] = 0
        regions, _, _ = segmentation.relabel_sequential(regions)
    # Determine rotation of embryos
    props_table = pd.DataFrame(measure.regionprops_table(
        regions, properties=['label', 'centroid']))
    centroids = np.array(props_table[["centroid-1", "centroid-0"]])
    theta = calculate_dish_rotation(centroids)
    rotated_im = np.array([transform.rotate(img[i], theta*180/np.pi,
                                            resize=False, preserve_range=True) for i in range(img.shape[0])])
    # Rotate and relabel segmented objects
    rotated_regions = transform.rotate(
        regions, theta*180/np.pi, cval=0, resize=False, order=0, preserve_range=True)
    rotated_props = pd.DataFrame(measure.regionprops_table(
        rotated_regions, properties=['label', 'bbox', 'centroid']))
    rotated_regions = remap_regions(rotated_regions, rotated_props)
    rotated_props = rotated_props.sort_values(
        by=["regularized_centroid-0", "centroid-1"])
    rotated_props["label"] = np.arange(rotated_props.shape[0])+1

    embryo_images = []
    processed_extra_arrays = []
    for i in range(rotated_props.shape[0]):
        row = rotated_props.iloc[i]
        if manual_bbox_size:
            end_y = min(
                int(row["bbox-0"] + manual_bbox_size[0]), rotated_im.shape[0])
            end_x = min(
                int(row["bbox-1"] + manual_bbox_size[1]), rotated_im.shape[1])
        else:
            end_y = int(row["bbox-2"])
            end_x = int(row["bbox-3"])
        cropped_im = rotated_im[:, int(
            row["bbox-0"]):end_y, int(row["bbox-1"]):end_x]
        if manual_bbox_size:
            pad_y = max(manual_bbox_size[0] - (end_y - int(row["bbox-0"])), 0)
            pad_x = max(manual_bbox_size[1] - (end_x - int(row["bbox-1"])), 0)
            # print(pad_y, pad_x)
            cropped_im = np.pad(cropped_im, np.array(
                [[0, 0], [0, pad_y], [0, pad_x]]))
            # print(cropped_im.shape)

        embryo_images.append(cropped_im)

    for arr in extra_split_arrays:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.imshow(arr)
        processed_extra_arrays.append([])
        if len(arr.shape) < 3:
            arr = np.expand_dims(arr, 0)

        rotated_array = np.array([transform.rotate(arr[i], theta*180/np.pi,
                                                   cval=0, preserve_range=True, resize=False) for i in range(arr.shape[0])])

        for i in range(rotated_props.shape[0]):
            row = rotated_props.iloc[i]
            processed_extra_arrays[-1].append(np.squeeze(rotated_array[:, int(row["bbox-0"]):int(row["bbox-2"]),
                                                                       int(row["bbox-1"]):int(row["bbox-3"])].astype(arr.dtype)))
    return embryo_images, rotated_props, rotated_regions, processed_extra_arrays


def translate_image(img, shift):
    """ Translate an image by a shift (x,y)
    """
    u, v = shift
    nr, nc = img.shape
    row_coords, col_coords = np.meshgrid(
        np.arange(nr), np.arange(nc), indexing='ij')
    return transform.warp(img, np.array([row_coords-v, col_coords-u]),
                          mode="constant", cval=np.nan)


def match_snap_to_data(img, ref, scale_factor=4):
    """ Match a snap of arbitrary size to an equally sized or smaller reference data set, assuming both ROIs are centered in camera coordinates
    """
    downscaled = transform.downscale_local_mean(
        img, (scale_factor, scale_factor))
    diff = (downscaled.shape[0] - ref.shape[0])//2
    if diff > 0:
        downscaled = downscaled[diff:-diff, diff:-diff]
    return downscaled
