""" Functions for loading and processing images.

"""
from pathlib import Path
from os import PathLike
import os
from typing import (
    Tuple,
    Union,
    Callable,
    TypeVar,
    Any,
    List,
    Dict,
    Collection,
    Optional,
    Iterable,
)
from collections.abc import Sequence
import pickle
import warnings

import numpy as np
from numpy import typing as npt

from scipy import signal, stats, interpolate, optimize, ndimage
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
from matplotlib import axes, figure

import skimage.io as skio
from skimage import (
    transform,
    filters,
    morphology,
    measure,
    draw,
    exposure,
    segmentation,
    feature,
)
from skimage.util import map_array
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from sklearn import cluster
import mat73
import pandas as pd

from . import traces
from . import stats as sstats
from .. import utils
from ..ui import visualize

MAX_INT32 = np.iinfo(np.int32).max
T = TypeVar("T")
NPGeneric = TypeVar("NPGeneric", bound=np.generic)


def regress_video(
    img: npt.NDArray, trace_array: npt.NDArray, regress_dc: bool = True
) -> npt.NDArray:
    """Linearly regress arbitrary traces from a video.

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
        data_matrix, trace_array, regress_dc=regress_dc
    ).reshape(img.shape)
    return regressed_video


def load_dmd_target(
    root_dir: Union[str, PathLike[Any]], expt_name: str, downsample_factor: int = 1
) -> npt.NDArray[np.bool_]:
    """Load the DMD target image from the experiment metadata.
    This is the image that the DMD is trying to project onto the screen.

    Args:
        root_dir: root directory of the experiment.
        expt_name: name of the experiment (single video).
        downsample_factor: factor by which to downsample the image. This deals with the fact that
            we downsample our images during processing.
    Returns:
        2D DMD target mask in image space.
    """
    # Load the .mat file containing the metadata.
    expt_data = mat73.loadmat(Path(root_dir, expt_name, "output_data_py.mat"))[
        "dd_compat_py"
    ]
    width = int(expt_data["camera"]["roi"][1])
    height = int(expt_data["camera"]["roi"][3])
    dmd_target = expt_data["dmd_lightcrafter"]["target_image_space"]

    # DMD transformation to image space is not perfect, so we need to crop the image to get rid of
    # the black border
    offset_x = int((dmd_target.shape[1] - width) / 2)
    offset_y = int((dmd_target.shape[0] - height) / 2)
    dmd_target = dmd_target[offset_y:-offset_y, offset_x:-offset_x]
    dmd_target = dmd_target[::downsample_factor, ::downsample_factor].astype(bool)
    return dmd_target


def load_raw_data(
    raw_dir: Union[str, PathLike[Any]],
    dims: Tuple[int, int],
    file_idx: int = 1,
    dtype_str: str = "<u2",
) -> npt.NDArray:
    """Load raw data from a .bin file. Handles the fact that the file name can be different.

    Args:
        raw_dir: directory containing the raw data.
        dims: dimensions of the raw data (rows, cols).
        file_idx: index of the file to load (if there are multiple files).
        dtype_str: data type of the raw data.
    Returns:
        3D array of raw data (time, rows, cols).
    Raises:
        FileNotFoundError: if the raw data file cannot be found.
    """
    rows, cols = dims
    rawpath = Path(raw_dir, "frames.bin")
    if not rawpath.exists():
        rawpath = Path(raw_dir, f"frames{file_idx+1}.bin")
    if not rawpath.exists():
        rawpath = Path(raw_dir, "Sq_camera.bin")
    if not rawpath.exists():
        raise FileNotFoundError(f"Could not find raw data file {str(rawpath)}")

    return np.fromfile(rawpath, dtype=np.dtype(dtype_str)).reshape((-1, rows, cols))


def load_tif_blocks(
    tif_dir: Union[str, PathLike[Any]],
    expt_name: str,
) -> npt.NDArray:
    """Load tif files in blocks to save memory.

    Args:
        tif_dir: directory containing the tif files.
        expt_name: name of the experiment broken up into blocks.
    Returns:
        3D array of tif data (time, x, y).
    """
    tif_files = sorted(Path(tif_dir).glob(f"{expt_name}*.tif"))
    blocks = []
    for tif_file in tif_files:
        blocks.append(skio.imread(tif_file))
    img = np.concatenate(blocks, axis=0)
    return img


def load_image(
    rootdir: Union[str, PathLike[Any]],
    expt_name: str,
    subfolder: str = "",
    raw: bool = True,
    cam_indices: Optional[Union[int, Iterable[int]]] = None,
    expt_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Union[npt.NDArray[np.generic], List[npt.NDArray[np.generic]]],
    Optional[Dict[str, Any]],
]:
    """General image loading function that handles various file formats floating around in
    Cohen lab.

    If raw is True, will try to load the raw data from the .bin file.  Otherwise,
    will try to load the tif files.  If there are multiple tif files, will concatenate them.

    Args:
        rootdir: root directory of the experiment.
        expt_name: name of the experiment (single video).
        subfolder: subfolder of the experiment (different stages of processing).
        raw: if True, will try to load the raw data from the .bin file.  Otherwise, will try
            to load the tif files.
        cam_indices: if there are multiple cameras, this is the index of the camera to load.
        expt_metadata: if the metadata has already been loaded, pass it in here
    Returns:
        Image data.  If there is only one camera, this will be a 3D array (time, x, y).  If there
            are multiple cameras, this will be a list of 3D arrays.
    """
    data_dir = Path(rootdir, subfolder)
    # Load the .mat file containing the metadata
    if expt_metadata is None:
        expt_metadata = utils.load_video_metadata(rootdir, expt_name)
    imgs = []

    if (
        raw and expt_metadata and (data_dir / expt_name).is_dir() and subfolder == ""
    ):  # Load binary images for each camera
        if cam_indices is None:
            cam_indices = range(len(expt_metadata["cameras"]))
        else:
            cam_indices = utils.make_iterable(cam_indices)
        for i in cam_indices:
            width = int(expt_metadata["cameras"][i]["roi"][1])
            height = int(expt_metadata["cameras"][i]["roi"][3])
            imgs.append(
                load_raw_data(data_dir / expt_name, (height, width), file_idx=i)
            )
    else:
        # Identify the number of tif files (files for each experiment not necessarily in their own
        # folder).
        imgs.append(load_tif_blocks(data_dir, expt_name))

    if expt_metadata and "frame_counter" in expt_metadata.keys():
        # Check to make sure that the number of frames in the .bin file matches the number of frames
        # expected from the metadata.
        fc_max = np.max(expt_metadata["frame_counter"])
        for i, img in enumerate(imgs):
            if fc_max > img.shape[0]:
                n_frames_dropped = fc_max - img.shape[0]
                warnings.warn(f"{n_frames_dropped} frames dropped")
                last_tidx = img.shape[0]
            else:
                last_tidx = int(
                    min(
                        fc_max,
                        expt_metadata["cameras"][i]["frames_requested"]
                        - expt_metadata["cameras"][i]["dropped_frames"],
                    )
                )
            imgs[i] = img[:last_tidx]
    # if isinstance(cam_indices
    if len(imgs) == 1:
        return imgs[0], expt_metadata
    return imgs, expt_metadata


def load_confocal_image(
    path: Union[str, PathLike], direction: str = "both", extra_offset: int = 2
) -> npt.NDArray:
    """Load a confocal image from a .mat file, because it is stored as a DAQ output trace.
    If direction is "fwd", will return the forward scan only.

    Args:
        path: path to the .mat file
        direction: "fwd" or "both"
        extra_offset: number of pixels to add to the shifting of reverse scan to align it with
            forward scan. This is a hack to deal with the fact that the reverse scan is shifted by a
            few pixels relative to the forward scan.
    Returns:
        confocal image (z-stack).
    """
    # Load the .mat file containing the metadata and extract information about the scan.
    matdata = mat73.loadmat(path)["dd_compat_py"]
    confocal = matdata["confocal"]
    points_per_line = int(confocal["points_per_line"])
    numlines = int(confocal["numlines"])

    # Turn output trace into an image
    img = confocal["PMT"].T.reshape(
        (confocal["PMT"].shape[1], numlines, 2, points_per_line)
    )

    # If direction is "fwd", return the forward scan only.
    if direction == "fwd":
        return img[:, :, 0, :]

    # If direction is "both", return the mean of the forward and reverse scans.
    elif direction == "both":
        fwd_scan = img[:, :, 0, :]
        rev_scan = img[:, :, 1, :]

        reshaped_xdata = confocal["xdata"].reshape(numlines, 2, points_per_line)
        pixel_step = reshaped_xdata[0, 0, 1] - reshaped_xdata[0, 0, 0]
        offset = np.min(
            np.mean(confocal["galvofbx"][: int(confocal["points_per_line"]), :], axis=1)
        )

        # Shift the reverse scan to align it with the forward scan.
        offset_pix = offset / pixel_step
        offset_idx = int(np.round(-(offset_pix + extra_offset)))
        revscan_shift = np.roll(np.flip(rev_scan, axis=2), offset_idx, axis=2)
        revscan_shift[:, :, :offset_idx] = np.mean(revscan_shift, axis=(1, 2))[
            :, np.newaxis, np.newaxis
        ]

        mean_img = (fwd_scan + revscan_shift) / 2
        return mean_img

    else:
        raise ValueError(f"direction = {direction} not implemented")


def extract_all_region_data(
    img: npt.NDArray, mask: npt.NDArray[np.unsignedinteger]
) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
    """Turn all mask regions into pixel-time traces of intensity

    Calls `extract_region_data` for each region in the mask.

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different
            region.
    Returns:
        region_data: list of 2D arrays (timepoints x pixels) describing intensities for each region.
        region_coords: list of 2D arrays (pixels x 2) describing the pixel coordinates for each
            region.
    """
    region_coords = []
    regions = np.unique(mask)
    regions = regions[regions >= 1]
    region_data = []

    for region in regions:
        rd, gc = extract_region_data(img, mask, region)
        region_coords.append(gc)
        region_data.append(rd)
    return region_data, region_coords


def extract_region_data(
    img: npt.NDArray, mask: npt.NDArray[np.unsignedinteger], region_index: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Turn raw image data from a specific region defined by integer-valued mask into a 2D matrix
    (timepoints x pixels)

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different
            region.
        region_index: integer value of the region to extract.
    Returns:
        region_data: 2D array (timepoints x pixels) for the specified region.
        global_coords: 2D array (pixels x 2) of the pixel coordinates for the specified region,
            relative to the whole image.

    """
    global_coords = np.argwhere(mask == region_index)
    region_data = np.zeros((img.shape[0], global_coords.shape[0]))
    # Loop over pixels in the region and extract the intensity values
    for px_idx in range(global_coords.shape[0]):
        px = global_coords[px_idx]
        region_data[:, px_idx] = img[:, px[0], px[1]]
    return region_data, global_coords


def extract_cropped_region_image(
    intensity: npt.NDArray, global_coords: Union[npt.NDArray, Tuple[int, int]]
) -> npt.NDArray:
    """Turn an unraveled list of intensities back into an image based on the
    bounding box of the specified global coordinates.

    Args:
        intensity (npt.ArrayLike): 1D array of intensity values or 2D array of intensity values over
            time (timepoints x pixels).
        global_coords (npt.NDArray): defined shape of image or 2D array of
            explicit global coordinates (pixels x 2).
    Returns:
        img (npt.NDArray): 2D array of intensity values.
    """
    if isinstance(global_coords, tuple):
        img = intensity.reshape(global_coords)
    else:
        if len(intensity.shape) > 2:
            raise TypeError("Expected 1D or 2D array of intensities")
        global_coords_rezeroed = global_coords - np.min(global_coords, axis=0)
        if len(intensity.shape) == 1:
            intensity = intensity[np.newaxis, :]

        img = np.zeros(
            [intensity.shape[0]] + list(np.max(global_coords_rezeroed, axis=0) + 1)
        )
        for idx in range(intensity.shape[1]):
            curr_px = global_coords_rezeroed[idx, :]
            img[:, curr_px[0], curr_px[1]] = intensity[:, idx]
        img = np.squeeze(img)
    return img


def extract_bbox_images(
    img: npt.NDArray, mask: npt.NDArray[np.unsignedinteger], padding: int = 0
) -> List[npt.NDArray]:
    """Get cropped images defined by the bounding boxes of ROIS provided by a mask

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a different
            region.
        padding: number of pixels to add to the bounding box on each side.
    Returns:
        cropped_images: list of 3D arrays (timepoints x pixels x pixels) for the bounding box
            defined by each region.
    """
    bboxes = [p["bbox"] for p in measure.regionprops(mask)]
    cropped_images = []
    for bbox in bboxes:
        r1 = max(bbox[0] - padding, 0)
        c1 = max(bbox[1] - padding, 0)
        r2 = min(bbox[2] + padding, img.shape[1])
        c2 = min(bbox[3] + padding, img.shape[2])
        cropped_images.append(img[:, r1:r2, c1:c2])
    return cropped_images


def extract_roi_traces(
    img: npt.NDArray, label_mask: npt.NDArray[np.unsignedinteger]
) -> npt.NDArray:
    """Get traces from image using defined ROI mask, where each ROI is defined by a unique integer
    value.

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        label_mask: 2D array of mask data (pixels x pixels). Each integer value corresponds to a
            different region.
    Returns:
        traces: 2D array of traces (ROIs x timepoints)
    """
    labels = np.unique(label_mask)
    labels = labels[labels != 0]
    image_traces = [extract_mask_trace(img, label_mask == l) for l in labels]
    return np.array(image_traces)


def extract_mask_trace(
    img: npt.NDArray, mask: Optional[npt.NDArray[np.bool_]] = None
) -> npt.NDArray:
    """Average part of an image to a trace according to a binary mask.

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mask: 2D array of mask data (pixels x pixels). If None, average over entire image.
    Returns:
        trace: 1D array of trace values (timepoints)
    """
    if mask is None:
        trace = img.mean(axis=(1, 2))
    elif len(mask.shape) == 3:
        masked_img: npt.NDArray = np.ma.masked_array(img, mask=~mask)
        trace = masked_img.mean(axis=(1, 2))
    else:
        # img is 2D, mask is 1D
        pixel_traces = img[:, mask]
        trace = pixel_traces.mean(axis=1)
    return trace


def background_subtract(img: npt.NDArray, dark_level: int = 100) -> npt.NDArray:
    """Syntactic sugar for background subtraction.

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        dark_level: value to subtract from each pixel
    Returns:
        img: 3D array of background subtracted image data (timepoints x pixels x pixels)
    """
    return img - dark_level


def extract_background_traces(
    img: npt.NDArray, mode: Union[str, Collection[str]] = "all", corner_divs: int = 5, 
    dark_percentile: int = 10, n_samps_localmin=101
) -> npt.NDArray:
    """Use one or more of several heuristics to find possible sources of background.

    Args:
        img: 3D array of raw image data (timepoints x pixels x pixels)
        mode: "all" or a list of strings. If "all", use all available methods. Otherwise, use only
            the methods specified in the list.
        corner_divs: number of divisions to make when calculating the mean of the corners
            of the image
        dark_percentile: The threshold to use for defining dark regions
    Returns:
        background_traces: 2D array of background traces (timepoints x methods)
    """
    if mode == "all":
        mode = ["linear", "mean", "dark", "corners", "exp", "biexp", "localmin"]
    background_traces: List[npt.NDArray] = []
    mean_trace = img.mean(axis=(1, 2))
    mean_img = img.mean(axis=0)
    tr: Union[npt.NDArray, List[npt.NDArray]]
    for m in mode:
        if m == "linear":
            tr = np.linspace(-1, 1, img.shape[0])
        elif m == "mean":
            tr = img.mean(axis=(1, 2))
        elif m in ("exponential", "exp"):
            _, tr, _ = traces.correct_photobleach(mean_trace, method="monoexp")
        elif m in ("biexponential", "biexp"):
            _, tr, _ = traces.correct_photobleach(mean_trace, method="biexp", b=3, a=5)
        elif m == "localmin":
            _, tr, _ = traces.correct_photobleach(mean_trace, method="localmin",
                                        nsamps=n_samps_localmin)
        elif m == "dark":
            mask = mean_img < np.percentile(mean_img, dark_percentile)
            tr = extract_mask_trace(img, mask)
        elif m == "corners":
            tr = []
            div0 = mean_img.shape[0] // corner_divs
            div1 = mean_img.shape[1] // corner_divs
            mask = np.zeros(mean_img.shape, dtype=bool)
            mask[:div0, :div1] = True
            tr.append(extract_mask_trace(img, mask))
            mask = np.zeros(mean_img.shape, dtype=bool)
            mask[-div0:, :div1] = True
            tr.append(extract_mask_trace(img, mask))
            mask = np.zeros(mean_img.shape, dtype=bool)
            mask[:div0, -div1:] = True
            tr.append(extract_mask_trace(img, mask))
            mask = np.zeros(mean_img.shape, dtype=bool)
            mask[-div0:, -div1:] = True
            tr.append(extract_mask_trace(img, mask))
        else:
            tr = np.zeros(img.shape[0])
        if isinstance(tr, np.ndarray):
            tr = [tr]
        background_traces.extend(tr)
    combined_traces = np.array(background_traces).T
    combined_traces -= combined_traces.mean(axis=0)
    combined_traces /= (combined_traces.max(axis=0) - combined_traces.min(axis=0))[
        None, :
    ]
    return combined_traces


def get_spike_kernel(img, kernel_length, nbefore, peak_prominence, savgol_length=51):
    """Estimate a temporal spike kernel by doing naive peak detection and selecting a temporal window.
    Pick a subsample with the largest peak amplitudes after smoothing (presumably largest SNR) and average.

    POSSIBLY UNUSED.

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
            trace_smoothed = signal.savgol_filter(img[:, i, j], savgol_length, 2)
            px_peaks, _ = signal.find_peaks(trace_smoothed, prominence=peak_prominence)
            if len(px_peaks) > 1:
                plt.plot(trace_smoothed)
                plt.plot(px_peaks, trace_smoothed[px_peaks], "rx")
                raise ValueError("More than one candidate peak detected")
            elif len(px_peaks) == 1:
                kernel_hits[i, j] = True
                pidx = px_peaks[0]
                if pidx < nbefore:
                    xs = np.arange(nbefore - pidx, kernel_length)
                    ys = img[: pidx - nbefore + kernel_length, i, j]
                    interpf = interpolate.interp1d(xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                elif pidx - nbefore + kernel_length > img.shape[0]:
                    xs = np.arange(img.shape[0] - (pidx - nbefore))
                    ys = img[pidx - nbefore :, i, j]
                    interpf = interpolate.interp1d(xs, ys, fill_value="extrapolate")
                    ktrace = interpf(np.arange(kernel_length))
                else:
                    ktrace = img[pidx - nbefore : pidx - nbefore + kernel_length, i, j]
                ktrace = (ktrace - np.min(ktrace)) / (
                    np.max(trace_smoothed) - np.min(trace_smoothed)
                )
                kernel_candidates.append(ktrace)
    kernel = np.zeros(img.shape[0])
    kernel[:kernel_length] = np.mean(np.array(kernel_candidates[:]), axis=0)
    kernel[kernel_length:] = kernel[kernel_length - 1]
    return kernel, kernel_hits


def snapt(img, kernel, offset_width=0):
    """Run SNAPT fitting algorithm (Hochbaum et al. Nature Methods 2014)
    POSSIBLY UNUSED
    """
    height = img.shape[1]
    width = img.shape[2]
    beta = np.zeros((height, width, 4))
    error_det = np.zeros((height, width))
    failed_counter = 0
    t0 = np.argmax(kernel)
    L = len(kernel)
    minshift = -t0
    maxshift = L - t0

    beta = np.apply_along_axis(
        lambda tr: kernel_fit_single_trace(
            tr, kernel, minshift, maxshift, offset_width
        ),
        0,
        img,
    )
    error_det = beta[4, :, :]
    beta = beta[:4, :, :]
    failed_pixels = np.sum(np.isnan(error_det))
    print(
        "%d/%d pixels failed to fit (%.2f %%)"
        % (failed_counter, height * width, failed_counter / (height * width) * 100)
    )
    return beta, error_det


def kernel_fit_single_trace(trace, kernel, minshift, maxshift, offset_width):
    """Nonlinear fit of empirical kernel to a single timeseries

    POSSIBLY UNUSED.

    """
    try:
        popt, pcov = optimize.curve_fit(
            utils.shiftkern,
            kernel,
            trace,
            p0=[1, 1, 1, np.random.randint(-offset_width, offset_width + 1)],
            absolute_sigma=True,
            bounds=([0, -np.inf, 0, minshift], [np.inf, np.inf, np.inf, maxshift]),
        )
        beta = popt
        error_det = np.linalg.det(pcov)
    except Exception as e:
        beta = utils.nans(4)
        error_det = np.nan
    beta = np.append(beta, error_det)
    return beta


def spline_fit_single_trace(
    trace: npt.ArrayLike,
    s: float,
    knots: Collection[float],
    plot: bool = False,
    n_iterations: int = 100,
    eps: float = 0.01,
    ax1: Optional[axes.Axes] = None,
) -> Union[
    Tuple[npt.NDArray[np.floating], interpolate.BSpline],
    Tuple[npt.NDArray[np.floating], interpolate.BSpline, axes.Axes],
]:
    """Least squares spline fitting of a single timeseries

    Args:
        trace: 1D array of timeseries
        s: smoothing parameter
        knots: number of knots to use for spline representation
        plot: whether to plot the fit
        n_iterations: number of iterations to run the fitting algorithm
        eps: convergence criterion for the fitting algorithm
        ax1: matplotlib axes to plot the fit on
    Returns:
        beta: spline features (absolute height, half-maximum time,relative amplitude, residual,
            time of maximum dx/dt)
        spl: spline representation of the trace
        (Optional) ax1: matplotlib axes with the plot
    """
    trace = np.array(trace)
    x = np.arange(len(trace))
    trace[np.isnan(trace)] = np.nanmin(trace)

    # Generate spline representation of the trace
    param_tck, res, _, _ = interpolate.splrep(
        x, trace, s=s, task=-1, t=knots, full_output=True, k=3
    )
    spl = interpolate.BSpline(*param_tck)
    spl_values = spl(x)
    dspl = spl.derivative()
    d2spl = spl.derivative(nu=2)

    extrema = np.argwhere(np.diff(np.sign(dspl(x)))).ravel()
    extrema_values = spl(extrema)

    # If no extrema are found within the trace, return NaNs
    if np.all(np.logical_or(extrema < 0, extrema > len(trace))):
        beta = utils.nans(5)
        return beta, spl

    # Find position and value of the global maximum
    global_max_index = np.argmax(extrema_values)
    global_max = extrema[global_max_index]
    global_max_val = extrema_values[global_max_index]
    min_val = np.nanpercentile(spl_values[:global_max], 10)
    halfmax_magnitude = (global_max_val - min_val) / 2 + min_val

    if plot:
        # Plot results
        if ax1 is None:
            _, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(trace)
        ax1.plot(spl(x))
        ax1.plot(extrema, extrema_values, "ko")
        ax1.axhline(float(min_val), color="r", linestyle="--")
        ax1.axhline(halfmax_magnitude, color="orange", linestyle="--")

    try:
        reversed_values = np.flip(spl_values[:global_max])
        # moving_away_from_hm = (np.diff(reversed_values - halfmax_magnitude) ** 2) > 0
        moving_away_from_hm = np.diff((reversed_values - halfmax_magnitude) ** 2) > 0

        # Estimate whether a particular index is close to the half-maximum by comparing the
        # squared difference between the value at that index and the half-maximum to the
        # squared difference between the global maximum and the minimum value.
        close_to_hm = (reversed_values - halfmax_magnitude) ** 2 < (
            (global_max_val - min_val) * 5 / global_max
        ) ** 2

        # Identify the index of the half-maximum by finding the first index where the value is both
        # close to the half-maximum and moving away from the half-maximum. This is done for
        # robustness to fluctuations in the trace that are small relative to the spike magnitude.
        hm = global_max - np.argwhere(close_to_hm[:-1] & moving_away_from_hm).ravel()[0]
        # Iterate to refine estimate
        hm0 = hm
        for _ in range(n_iterations):
            hm1 = utils.custom_newton_lsq(
                hm0, halfmax_magnitude, spl, dspl, bounds=(hm - 3, hm + 3)
            )
            if (hm1 - hm0) ** 2 < eps**2:
                break
            hm0 = hm1
        halfmax = hm0

        # Look for sign changes in the second derivative of the spline to identify the position
        # of the maximum first derivative (another method of identifying wavefronts).
        local_max_deriv_idx = np.argwhere(np.diff(np.sign(d2spl(x)))).ravel()
        local_max_deriv_idx = local_max_deriv_idx[local_max_deriv_idx < global_max]
        if len(local_max_deriv_idx) > 0:
            max_deriv_idx = local_max_deriv_idx[
                np.argmin((local_max_deriv_idx - halfmax) ** 2)
            ]
            max_deriv_interp = max_deriv_idx + (0 - d2spl(max_deriv_idx)) / (
                d2spl(max_deriv_idx + 1) - d2spl(max_deriv_idx)
            )
        else:
            max_deriv_interp = np.nan
    except IndexError:
        beta = utils.nans(5)
        return beta, spl

    beta = np.array(
        [global_max, halfmax, global_max_val / min_val, res, max_deriv_interp]
    )

    if plot and ax1:
        ax1.plot(halfmax, spl(halfmax), "gx")
        ax1.plot(max_deriv_interp, spl(max_deriv_interp), "bx")
        return beta, spl, ax1
    else:
        return beta, spl


def spline_timing(
    img: npt.NDArray, s: float = 0.1, n_knots: int = 4, upsample_rate: float = 1
):
    """Perform spline fitting to functional imaging data do determine wavefront timing

    Args:
        img : 3D array of functional imaging data (time x y x z)
        s : smoothing parameter for spline fitting
        n_knots : number of knots to use for spline fitting
        upsample_rate : factor by which to upsample the video
    Returns:
        beta : 3D array of fit parameters (6 x y x z): absolute height, half-maximum time, relative
            amplitude, residual, time of maximum dx/dt, temporal noise
        smoothed_vid : 3D array of smoothed video (time x y x z)
    """
    knots = np.linspace(0, img.shape[0] - 1, num=n_knots)[1:-1]
    q = np.apply_along_axis(lambda tr: spline_fit_single_trace(tr, s, knots), 0, img)
    # Convert fit parameters in beta to images
    beta = np.moveaxis(
        np.array(list(q[0].ravel())).reshape((img.shape[1], img.shape[2], -1)), 2, 0
    )
    # Smooth the video using the spline fits and optionally upsample
    x = np.arange(img.shape[0] * upsample_rate) / upsample_rate
    smoothed_vid = np.array([spl(x) for spl in q[1].ravel()])
    smoothed_vid = np.moveaxis(
        smoothed_vid.reshape((img.shape[1], img.shape[2], -1)), 2, 0
    )
    noise_estimate = np.std(img - smoothed_vid, axis=0)
    beta = np.concatenate([beta, noise_estimate[np.newaxis, :, :]], axis=0)
    return beta, smoothed_vid


def process_isochrones(
    beta: npt.NDArray[np.floating],
    dt: float,
    threshold: Optional[float] = None,
    plot: bool = False,
    intensity_mask: Optional[npt.NDArray[np.bool_]] = None,
    threshold_mode: str = "amplitude",
    med_filt_size: int = 3,
    opening_size: int = 3,
    closing_size: int = 3,
    amplitude_artifact_cutoff: float = 2.5,
    dilation_size: int = 0,
    valid_mask: Optional[npt.NDArray[np.bool_]] = None,
    t_pctile_cutoffs = (0,100)
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Clean up spline fitting to better visualize isochrones: get rid of nans and low-amplitude
    values

    Args:
        beta : 3D array of fit parameters (6 x y x z): absolute height, half-maximum time,
            relative amplitude, residual, time of maximum dx/dt, temporal noise
        dt : time between frames
        threshold : threshold for spike amplitude to determine acceptable pixels. If None,
            threshold is determined automatically.
        plot : whether to plot the results
        intensity_mask : Optional mask of pixels with sufficient raw intensity.
        threshold_mode : "amplitude" or "snr"
        med_filt_size : size of median filter to apply to isochrones.
        opening_size : size of opening filter to apply to mask.
        closing_size : size of closing filter to apply to mask.
        amplitude_artifact_cutoff : max amplitude cutoff.
        dilation_size : size of dilation filter to apply to mask.
        valid_mask : Optional manual mask of acceptable pixels.
    Returns:
        hm_smoothed: 3D array of smoothed half-maximum times (y x z) in milliseconds
        dv_smoothed: 3D array of smoothed activation times based on max temporal derivative in
            milliseconds
    Raises:
        ValueError: if threshold_mode is invalid
    """
    print(threshold_mode)
    if threshold_mode == "amplitude":
        amplitude = beta[2, :, :]
    elif threshold_mode == "snr":  # calculate SNR (power)
        amplitude = (np.abs(beta[2] - 1) / beta[5]) ** 2
    else:
        raise ValueError("Invalid threshold_mode")
    isochron_mask: npt.NDArray[np.bool_]
    if valid_mask is None:  # if no valid_mask is provided, calculate it automatically
        amplitude_nanr: npt.NDArray[np.floating] = np.copy(amplitude)
        # Remove outlier values and NaNs
        amplitude_nanr[beta[2] > amplitude_artifact_cutoff] = np.nan
        amplitude_nanr[np.isnan(amplitude_nanr)] = np.nanmin(amplitude)
        amplitude_nanr = ndimage.gaussian_filter(amplitude_nanr, sigma=3).astype(float)
        if threshold is None:
            # If not provided, determine threshold automatically
            threshold = min(max(filters.threshold_triangle(amplitude_nanr), 2), 100)
        if intensity_mask is None:
            # If not provided, do not apply intensity mask
            intensity_mask = np.ones_like(amplitude, dtype=bool)

        mask = (amplitude_nanr > threshold) & intensity_mask
        fig1, ax1 = plt.subplots(figsize=(4,4))
        ax1.imshow(mask)
        if opening_size > 0:
            mask = morphology.binary_opening(
                mask, footprint=morphology.disk(opening_size)
            )
        if closing_size > 0:
            mask = morphology.binary_closing(
                mask, footprint=morphology.disk(closing_size)
            )

        # Find the largest connected component
        labels = measure.label(mask)
        label_values, counts = np.unique(labels, return_counts=True)
        label_values = label_values[1:]
        counts = counts[1:]
        try:
            isochron_mask = labels == label_values[np.argmax(counts)]
            if dilation_size > 0:
                isochron_mask = morphology.binary_dilation(
                    isochron_mask, footprint=morphology.disk(dilation_size)
                )
        except ValueError:
            isochron_mask = np.zeros_like(amplitude, dtype=bool)

        if plot:
            _, ax1 = plt.subplots(figsize=(3, 3))
            visualize.display_roi_overlay(amplitude_nanr, mask.astype(int), ax=ax1)
    else:
        isochron_mask = valid_mask

    hm = beta[1, :, :]
    dv = beta[4, :, :]
    t_cutoff_hm = np.nanpercentile(hm, t_pctile_cutoffs)
    t_cutoff_dv = np.nanpercentile(hm, t_pctile_cutoffs)
    hm[(hm<t_cutoff_hm[0])|(hm>t_cutoff_hm[1])] = np.nan
    dv[(dv<t_cutoff_dv[0])|(dv>t_cutoff_dv[1])] = np.nan
    # Replace NaNs with the mean of valid surrounding pixels
    hm_nans_removed = remove_nans(hm, kernel_size=13)
    dv_nans_removed = remove_nans(dv, kernel_size=13)
    # Smooth the isochrones
    hm_smoothed: npt.NDArray[np.floating] = ndimage.median_filter(
        hm_nans_removed, size=med_filt_size
    ).astype(float)
    hm_smoothed = ndimage.gaussian_filter(hm_smoothed, sigma=1).astype(float)
    hm_smoothed[~isochron_mask] = np.nan
    hm_smoothed *= dt * 1000

    dv_smoothed: npt.NDArray[np.floating] = ndimage.median_filter(
        dv_nans_removed, size=med_filt_size
    ).astype(float)
    dv_smoothed = ndimage.gaussian_filter(dv_smoothed, sigma=1).astype(float)
    dv_smoothed[~isochron_mask] = np.nan
    dv_smoothed *= dt * 1000

    return hm_smoothed, dv_smoothed


def clamp_intensity(
    img: npt.NDArray[np.floating], pctiles: Tuple[float, float] = (2.0, 99.0)
) -> npt.NDArray[np.floating]:
    """ Clamp intensities to the given percentiles
    Args:
        img: 2D image
        pctiles: Percentiles to use for clamping
    Returns:
        processed_img: 2D image with values clamped to the given percentiles
    """
    min_val, max_val = np.nanpercentile(img, [*pctiles])
    processed_img = np.copy(img)
    processed_img[processed_img > max_val] = max_val
    processed_img[processed_img < min_val] = min_val
    return processed_img


def normalize_and_clamp(
    img: npt.NDArray[np.floating], pctiles: Tuple[float, float] = (0, 99)
) -> npt.NDArray[np.floating]:
    """ Normalize an image and clamp values to the given percentiles

    Args:
        img: 2D image
        pctiles: Percentiles to use for clamping. If the first value is None, the minimum value is
            set to 0.
    Returns:
        processed_img: 2D image with values clamped to the given percentiles
    """
    min_val, max_val = np.nanpercentile(img, [*pctiles])
    processed_img = (img - min_val) / (max_val - min_val)
    processed_img[processed_img > 1] = 1
    processed_img[processed_img < 0] = 0
    return processed_img


def remove_nans(
    img: npt.NDArray[np.floating], kernel_size: int = 3
) -> npt.NDArray[np.floating]:
    """Replace NaNs in a 2D image with an average of surrounding values

    Args:
        img: 2D image
        kernel_size: Size of the kernel to use for averaging
    Returns:
        nans_removed: 2D image with NaNs replaced
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    nans_removed = np.copy(img)
    nans_removed[np.isnan(img)] = ndimage.generic_filter(
        img,
        np.nanmean,
        footprint=np.ones((kernel_size, kernel_size)),
        mode="constant",
        cval=np.nan,
    )[np.isnan(img)]
    return nans_removed


def estimate_local_velocity(
    activation_times: npt.NDArray[np.floating],
    deltax: int = 7,
    deltay: int = 7,
    deltat: float = 100,
    valid_points_threshold: int = 10,
    debug: bool = False,
    weights: Optional[npt.NDArray[np.floating]] = None,
) -> Union[
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]],
]:
    """Use local polynomial fitting strategy from Bayley et al. 1998 to determine velocity from
    activation map

    Args:
        activation_times: 2D array of activation times
        deltax: Size of local region in x direction
        deltay: Size of local region in y direction
        deltat: Time interval for local region
        valid_points_threshold: Minimum number of points required to fit a local polynomial
        debug: Whether to plot the results
        weights: Optional weights to use for local polynomial fitting
    Returns:
        v: 3D array of velocity vectors
        t_smoothed: 2D array of smoothed activation times
        n_points_fit: 2D array of number of points used for local polynomial fitting at each pixel
    """
    X, Y = np.meshgrid(
        np.arange(activation_times.shape[1]), np.arange(activation_times.shape[0])
    )
    coords = np.array([Y.ravel(), X.ravel()]).T
    t_smoothed = np.empty_like(activation_times)
    residuals = np.empty_like(activation_times)
    t_smoothed = utils.nans_like(activation_times)
    residuals = utils.nans_like(activation_times)
    # Note v = (v_y, v_x), consistent with row and column convention
    v = utils.nans((2, activation_times.shape[0], activation_times.shape[1]))
    n_points_fit = np.zeros_like(activation_times)

    for y, x in coords:
        # Generate local coordinates
        local_x, local_y = (
            np.meshgrid(
                np.arange(
                    max(0, x - deltax), min(activation_times.shape[1], x + deltax + 1)
                ),
                np.arange(
                    max(0, y - deltay), min(activation_times.shape[0], y + deltay + 1)
                ),
            )
            - np.array([x, y])[:, np.newaxis, np.newaxis]
        )
        # Generate local activation times
        local_times = activation_times[
            max(0, y - deltay) : min(activation_times.shape[0], y + deltay + 1),
            max(0, x - deltax) : min(activation_times.shape[1], x + deltax + 1),
        ]
        local_x = local_x.ravel()
        local_y = local_y.ravel()
        b = local_times.ravel()

        # Generate local design matrix
        A = np.array(
            [
                local_x**2,
                local_y**2,
                local_x * local_y,
                local_x,
                local_y,
                np.ones_like(local_x),
            ]
        ).T
        # Apply optional weights
        if weights is None:
            w = np.ones((A.shape[0], 1))
        else:
            w = weights[
                max(0, y - deltay) : min(activation_times.shape[0], y + deltay + 1),
                max(0, x - deltax) : min(activation_times.shape[1], x + deltax + 1),
            ].ravel()[:, None]
        # Remove NaNs
        w = w[~np.isnan(b), :]
        A = A[~np.isnan(b), :]
        b = b[~np.isnan(b)]
        # Remove points outside of time window
        mask = np.abs(b - activation_times[y, x]) < deltat
        w = w[mask, :]
        A = A[mask, :] * np.sqrt(w)
        b = b[mask] * np.sqrt(w).ravel()

        # Check if there are enough points to fit
        n_points_fit[y, x] = len(b)
        if n_points_fit[y, x] < valid_points_threshold:
            continue
        try:
            # Fit polynomial of degree 2 to coordinates and activation times
            p, res, _, _ = np.linalg.lstsq(A, b)
            t_smoothed[y, x] = p[5]
            residuals[y, x] = res
            # To make v = (v_y, v_x, flip)
            v[:, y, x] = np.flip(p[3:5] / (p[3] ** 2 + p[4] ** 2))
        except (np.linalg.LinAlgError, ValueError):
            pass
    if debug:
        return v, t_smoothed, n_points_fit
    else:
        return v, t_smoothed


def correct_photobleach(
    img: npt.NDArray[Union[np.floating, np.integer]],
    mask: Union[npt.NDArray[np.bool_], None] = None,
    method: str = "localmin",
    nsamps: int = 51,
    amplitude_window: float = 0.5,
    dt: float = 0.01,
    invert: bool = False,
    return_params: bool = False,
) -> Union[
    npt.NDArray[np.floating],
    Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]],
]:
    """Perform photobleach correction on each pixel in an image

    Args:
        img: 3D array of fluorescence traces (n_frames, n_rows, n_cols)
        mask: 2D array of pixels to select for generating photobleaching trace
        method: Method to use for photobleach correction. Options are:
            "linear": Perform linear fit to each pixel independently over time
            "localmin": Correct against moving minimum of masked average
            "monoexp": Correct against monoexponential fit to masked average
            "biexp": Correct against biexponential fit to masked average
            "decorrelate": Remove masked average over time
        nsamps: Number of samples to use for localmin method
        amplitude_window: Portion of trace (in real time) to use exponential amplitude
            estimation (biexp and monoexp methods)
        dt: Time step for exponential fit (biexp and monoexp methods)
        invert: Invert the image before performing photobleach correction (e.g. for downward-going
            spikes)
        return_params: Return parameters of exponential fit (biexp and monoexp methods)
    Returns:
        corrected_img: 3D array of corrected fluorescence traces
        (optional) photobleach_trace: 1D array of photobleach trace
        (optional) params: 3D array of exponential fit parameters
    """
    if method == "linear":
        # Perform linear fit to each pixel independently over time
        pixel_traces = img.reshape(img.shape[0], -1)
        A = np.array([np.arange(img.shape[0]), np.ones(img.shape[0])]).T
        x = np.linalg.pinv(A) @ pixel_traces
        regressed_traces = pixel_traces - x[0] * np.arange(img.shape[0])[:, np.newaxis]
        corrected_img = regressed_traces.reshape(img.shape)

    elif method == "localmin":
        # Correct against moving minimum of masked average
        if invert:
            # for negative-going spikes (e.g. Voltron)
            mean_img = img.mean(axis=0)
            raw_img = 2 * mean_img - img
        else:
            raw_img = img
        mean_trace = extract_mask_trace(raw_img, mask)
        # Get photobleaching trace for the average of the masked area
        _, pbleach, _ = traces.correct_photobleach(
            mean_trace, method=method, nsamps=nsamps
        )
        # Divide each pixel by the photobleaching trace (assuming each pixel has the same
        # photobleaching profile)
        corrected_img = np.divide(raw_img, pbleach[:, np.newaxis, np.newaxis])
        if return_params:
            return corrected_img, pbleach, np.array([])

    elif method == "monoexp" or method == "biexp":
        # Correct against a exponential decay of fluorescence
        mean_trace = extract_mask_trace(img, mask)
        background_level = np.percentile(img, 5, axis=0)

        # Get photobleaching trace for the average of the masked area
        if method == "monoexp":
            _, pbleach, params = traces.correct_photobleach(
                mean_trace,
                method="monoexp",
                return_params=True,
                invert=invert,
                a=5,
                b=2,
            )
        elif method == "biexp":
            _, pbleach, params = traces.correct_photobleach(
                mean_trace,
                method="biexp",
                return_params=True,
                invert=invert,
                a=5,
                b=2,
            )
        # It's too computationally expensive to fit a monoexponential decay to each pixel, so this
        # is a way to roughly estimate the amplitude of the decay. We use extreme values observed in
        # the trace to estimate the amplitude, and then apply a median filter to smooth it out.
        dur = img.shape[0]
        amplitude_window_idx = int(amplitude_window / dt)
        max_val = np.median(img[:amplitude_window_idx], axis=0)
        min_val = np.median(img[-amplitude_window_idx:], axis=0)
        amplitude = (max_val - min_val) / (1 - np.exp(params[1] * dur))
        amplitude[~np.isfinite(amplitude)] = 0
        amplitude = np.maximum(amplitude, 0)
        amplitude = filters.median(amplitude, selem=morphology.disk(7))
        amplitude[amplitude / params[0] * pbleach[-1] > background_level] = 0

        # Correct each pixel by the photobleaching trace (only if the amplitude is large enough)
        corrected_img = (
            img
            - np.divide(
                amplitude,
                params[0],
                out=np.zeros_like(amplitude),
                where=params[0] > 5e-2,
            )
            * pbleach[:, np.newaxis, np.newaxis]
        )

        if return_params:
            return corrected_img, amplitude, params
    elif method == "decorrelate":
        # Perform least-squares regression to remove the spatial mean over time
        mean_trace = img.mean(axis=(1, 2))
        corrected_img = regress_video(img, mean_trace)
        if return_params:
            return corrected_img, mean_trace, np.array([])
    else:
        raise ValueError("Not Implemented")

    return corrected_img


def get_image_dFF(
    img: npt.NDArray[Union[np.floating, np.integer]],
    baseline_percentile: float = 10,
    t_range: Tuple[int, int] = (0, -1),
    invert: bool = False,
) -> npt.NDArray[np.floating]:
    """Convert a raw image into dF/F

    Args:
        img: 3D array of raw image data
        baseline_percentile: percentile of time range to use for baseline
        t_range: time range to use for baseline
        invert: whether to invert the image (for negative-going voltage indicators)
    Returns:
        dFF: 3D array of dF/F image data
    """
    if invert:
        img = 2 * np.mean(img, axis=0) - img
    baseline = np.percentile(img[t_range[0] : t_range[1]], baseline_percentile, axis=0).astype(np.float32)
    dFF = img / baseline
    return dFF


def get_spike_videos(
    img: npt.NDArray[Union[np.floating, np.integer]],
    peak_indices: Union[npt.NDArray[np.integer], Sequence[int]],
    bounds: Tuple[int, int],
    normalize_height: bool = True,
):
    """Generate spike-triggered videos of a defined length from a recording and known peak indices

    Args:
        img: 3D array of raw image data (time x height x width)
        peak_indices: indices of peaks to use for spike-triggered videos
        bounds: time bounds to use for spike-triggered videos (before, after)
        normalize_height: whether to normalize the height of each spike-triggered video
    Returns:
        spike_imgs: 4D array of spike-triggered videos (spike x time x height x width)
    """
    before, after = bounds
    spike_imgs = utils.nans(
        (len(peak_indices), before + after, img.shape[1], img.shape[2])
    )

    for pk_idx, pk in enumerate(peak_indices):
        start, end = max(0, pk - before), min(img.shape[0], pk + after)
        spike_img = np.pad(
            img[start:end, :, :],
            ((max(before - pk, 0), max(pk + after - img.shape[0], 0)), (0, 0), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
        if normalize_height:
            spike_img /= np.nanpercentile(spike_img, 99)
        spike_imgs[pk_idx] = spike_img
    return spike_imgs


def spike_triggered_average_video(
    img: npt.NDArray[Union[np.floating, np.integer]],
    peak_indices: npt.NDArray[np.integer],
    sta_bounds: Tuple[int, int],
    include_mask: Optional[npt.NDArray[np.bool_]] = None,
    normalize_height: bool = False,
    full_output: bool = False,
) -> Tuple[npt.NDArray[np.floating], Optional[npt.NDArray[np.floating]]]:
    """Create a spike-triggered average video from a recording and known peak indices

    Args:
        img: 3D array of raw image data (time x height x width)
        peak_indices: 1D array of peak indices
        sta_bounds: time bounds for spike-triggered average (before, after)
        include_mask: boolean mask of which peaks to include
        normalize_height: whether to normalize the height of each event
        full_output: whether to return the full output (including videos of each event)
    Returns:
        sta: 2D array of spike-triggered average image
        spike_triggered_images: 4D array of spike-triggered images (if full_output=True)

    """
    if include_mask is None:
        include_mask = np.ones_like(peak_indices, dtype=bool)
    spike_triggered_images = get_spike_videos(
        img,
        peak_indices[include_mask],
        sta_bounds,
        normalize_height=normalize_height,
    )
    sta = np.nanmean(spike_triggered_images, axis=0)
    if full_output:
        return sta, spike_triggered_images
    else:
        return sta, None

def analyze_wave_dynamics(
    beta,
    dt,
    um_per_px,
    mask_function_tsmoothed=None,
    deltax=9,
    deltat=350,
    threshold_mode="snr",
    **isochrone_process_params,
):
    """Measure spatiotemporal properties of wave propagation
    
    Args:
        beta: 6-element array of beta parameters
        dt: time step of data
        um_per_px: microns per pixel
        mask_function_tsmoothed: function to use for masking activation map to throw out bad regions
            of video
        deltax: spatial window size for local smoothing according to Bayly et al. 1999
        deltat: temporal window size for local smoothing according to Bayly et al. 1999
        isochrone_process_params: parameters to pass to process_isochrones
    Returns:
        results: tuple of (mean velocity, median velocity, LOI position according to halfmax xy, 
            LOI position according to maximum upstroke velocity xy)
        Tsmoothed: smoothed activation map according to halfmax method
        Tsmoothed_dv: smoothed activation map according to maximum upstroke velocity method
        divergence: divergence of velocity field
        v: velocity field
    """

    def default_mask(Ts, divergence):
        nan_mask = morphology.binary_dilation(
            np.pad(np.isnan(Ts), 1, constant_values=True), selem=morphology.disk(3)
        )

        return np.ma.masked_array(Ts, nan_mask[1:-1, 1:-1] | (divergence < 0.5))

    if mask_function_tsmoothed is None:
        mask_function_tsmoothed = default_mask
    hm_nan, dv_max_nan = process_isochrones(
        beta, dt, threshold_mode=threshold_mode, **isochrone_process_params
    )
    if np.sum(~np.isnan(hm_nan)) == 0:
        return None
    snr = np.nan_to_num((np.abs(beta[2] - 1) / beta[5]) ** 2)
    _, Tsmoothed = estimate_local_velocity(
        hm_nan, deltax=deltax, deltay=deltax, deltat=deltat, weights=snr
    )
    v, Tsmoothed_dv = estimate_local_velocity(
        dv_max_nan, deltax=deltax, deltay=deltax, deltat=deltat, weights=snr
    )
    v *= um_per_px * 1000
    abs_vel = np.linalg.norm(v, axis=0)
    mean_velocity = np.nanmean(abs_vel.ravel())
    median_velocity = np.nanmedian(abs_vel.ravel())
    divergence = utils.div(v / abs_vel)

    masked_tsmoothed = mask_function_tsmoothed(Tsmoothed, divergence)
    masked_tsmoothed_dv = mask_function_tsmoothed(Tsmoothed_dv, divergence)

    min_activation_time = np.unravel_index(
        np.ma.argmin(masked_tsmoothed), Tsmoothed.shape
    )
    min_activation_time_dv = np.unravel_index(
        np.ma.argmin(masked_tsmoothed_dv), Tsmoothed_dv.shape
    )

    results = (
        mean_velocity,
        median_velocity,
        min_activation_time[1],
        min_activation_time[0],
        min_activation_time_dv[1],
        min_activation_time_dv[0],
    )
    return results, Tsmoothed, Tsmoothed_dv, divergence, v


def downsample_video(
    raw_img: npt.NDArray,
    downsample_factor: int,
    aa: Union[str, None] = "gaussian",
) -> npt.NDArray:
    """Downsample video in space (in integer increments) with optional anti-aliasing"""
    if downsample_factor == 1:
        di = raw_img
    else:
        if aa == "butter":
            sos = signal.butter(4, 1 / downsample_factor, output="sos")
            smoothed: npt.NDArray = np.apply_over_axes(
                lambda a, axis: np.apply_along_axis(
                    lambda x: signal.sosfiltfilt(sos, x), axis, a
                ),
                raw_img,
                [1, 2],
            )
            di = smoothed[:, ::downsample_factor, ::downsample_factor]
        elif aa == "gaussian" and raw_img.dtype != np.bool_:
            smoothed: npt.NDArray = ndimage.gaussian_filter(
                raw_img, [0, (downsample_factor - 1) / 2, (downsample_factor - 1) / 2]
            )
            di = smoothed[:, ::downsample_factor, ::downsample_factor]
        else:
            di = transform.downscale_local_mean(
                raw_img, (1, downsample_factor, downsample_factor)
            )
    return di


def get_image_dff_corrected(img, nsamps_corr, mask=None, plot=None, full_output=False):
    """Convert image to Delta F/F after doing photobleach correction and sampling"""

    # Generate mask if not provided
    mean_img = img.mean(axis=0)
    if mask is None:
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0] / 50)
        mask = morphology.binary_closing(
            mask, selem=np.ones((kernel_size, kernel_size))
        )

    if plot is not None:
        _, _ = visualize.display_roi_overlay(mean_img, mask.astype(int), ax=plot)
    # Correct for photobleaching and convert to DF/F
    pb_corrected_img = correct_photobleach(
        img, mask=np.tile(mask, (img.shape[0], 1, 1)), nsamps=nsamps_corr
    )
    dFF_img = get_image_dFF(pb_corrected_img)

    if full_output:
        return dFF_img, mask
    else:
        return dFF_img


def image_to_peaks(
    img,
    mask=None,
    prom_pct=90,
    exclude_pks=None,
    fs=1,
    min_width_s=0.5,
    f_c=2.5,
    **peak_detect_params,
):
    """Detect spike events in image"""

    if mask is None:
        mean_img = img.mean(axis=0)
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0] / 50)
        mask = morphology.binary_closing(
            mask, selem=np.ones((kernel_size, kernel_size))
        )

    sos = signal.butter(5, f_c, btype="lowpass", output="sos", fs=fs)

    mask_trace = extract_mask_trace(img, mask=np.tile(mask, (img.shape[0], 1, 1)))
    mask_trace = signal.sosfiltfilt(sos, mask_trace)
    ps = np.percentile(mask_trace, [10, prom_pct])
    prominence = ps[1] - ps[0]

    pks, _ = signal.find_peaks(
        mask_trace,
        prominence=prominence,
        width=(int(min_width_s * fs), None),
        rel_height=0.8,
        **peak_detect_params,
    )

    if exclude_pks:
        keep = np.ones(len(pks), dtype=bool)
        keep[exclude_pks] = 0
        pks = pks[keep]

    return pks, mask_trace


def image_to_sta(
    raw_img,
    fs=1,
    mask=None,
    plot=False,
    savedir=None,
    prom_pct=90,
    sta_bounds="auto",
    exclude_pks=None,
    offset=0,
    normalize_height=True,
    full_output=False,
    min_width_s=0.5,
):
    """Convert raw image into spike triggered average"""
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    # Generate mask if not provided
    mean_img = raw_img.mean(axis=0)
    if mask is None:
        mask = mean_img > np.percentile(mean_img, 80)
        kernel_size = int(mask.shape[0] / 50)
        mask = morphology.binary_closing(
            mask, footprint=np.ones((kernel_size, kernel_size))
        )
    raw_trace = extract_mask_trace(raw_img, np.tile(mask, (raw_img.shape[0], 1, 1)))

    # convert to DF/F

    dFF_img = get_image_dFF(raw_img)
    pks, dFF_mean = image_to_peaks(
        dFF_img - 1,
        mask=mask,
        prom_pct=prom_pct,
        fs=fs,
        exclude_pks=exclude_pks,
        min_width_s=min_width_s,
    )

    if savedir:
        skio.imsave(os.path.join(savedir, "dFF.tif"), dFF_img.astype(np.float32))
        skio.imsave(
            os.path.join(savedir, "dFF_display.tif"),
            exposure.rescale_intensity(dFF_img, out_range=(0, 255)).astype(np.uint8),
        )

    dFF_mean = extract_mask_trace(dFF_img, mask=np.tile(mask, (dFF_img.shape[0], 1, 1)))

    if plot:
        fig1, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.ravel()
        # Show mean image and mask used to aquire trace
        _ = visualize.display_roi_overlay(mean_img, mask.astype(int), ax=axs[0])

        # Plot mask trace of dff and raw
        axs[1].plot(np.arange(dFF_img.shape[0]) / fs, dFF_mean, color="C2")
        axs[1].plot(pks / fs, dFF_mean[pks], "rx")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel(r"$F/F_0$")
        tx = axs[1].twinx()
        tx.plot(np.arange(dFF_img.shape[0]) / fs, raw_trace, color="C1")
        tx.set_ylabel("Mean counts")
    else:
        axs = None

    # Automatically determine bounds for spike-triggered average
    if sta_bounds == "auto":
        try:
            # Align all detected peaks of the full trace and take the mean
            aligned_traces = traces.align_fixed_offset(
                np.tile(dFF_mean, (len(pks), 1)), pks
            )
        except Exception as e:
            print(e)
            return raw_trace
        mean_trace = np.nanmean(aligned_traces, axis=0)
        # Smooth
        spl = interpolate.UnivariateSpline(
            np.arange(len(mean_trace)),
            np.nan_to_num(mean_trace, nan=np.nanmin(mean_trace)),
            s=0.001,
        )
        smoothed = spl(np.arange(len(mean_trace)))
        smoothed_dfdt = spl.derivative()(np.arange(len(mean_trace)))
        # Find the first minimum before and the first minimum after the real spike-triggered peak
        try:
            minima_left = np.argwhere((smoothed_dfdt < 0)).ravel()
            minima_right = np.argwhere((smoothed_dfdt > 0)).ravel()
            # calculate number of samples to take before and after the peak
            b1 = pks[-1] - minima_left[minima_left < pks[-1]][-1] + offset
            b2 = minima_right[minima_right > pks[-1]][0] - pks[-1] + offset

            if plot:
                assert axs is not None
                axs[3].plot(mean_trace)
                axs[3].plot(smoothed, color="C1")
                tx = axs[3].twinx()
                tx.plot(smoothed_dfdt, color="C2")
                tx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                axs[3].plot(
                    [pks[-1] - b1, pks[-1], pks[-1] + b2],
                    mean_trace[[pks[-1] - b1, pks[-1], pks[-1] + b2]],
                    "rx",
                )

        except Exception:
            return mean_trace
        sta_bounds = (b1, b2)

    # Collect spike traces according to bounds
    sta_trace = traces.get_sta(
        dFF_mean, pks, sta_bounds, f_s=fs, normalize_height=normalize_height
    )

    if plot:
        axs[2].plot(
            (np.arange(sta_bounds[0] + sta_bounds[1]) - sta_bounds[0]) / fs, sta_trace
        )
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel(r"$F/F_0$")
        axs[2].set_title(f"STA taps: {sta_bounds[0]} + {sta_bounds[1]}")
        plt.tight_layout()
    # Get STA video

    sta, spike_images = spike_triggered_average_video(
        dFF_img,
        pks,
        sta_bounds,
        normalize_height=normalize_height,
        full_output=full_output,
    )

    if savedir:
        skio.imsave(os.path.join(savedir, "sta.tif"), sta)
        if plot:
            plt.savefig(os.path.join(savedir, "QA_plots.tif"))
        with open(os.path.join(savedir, "temporal_statistics.pickle"), "wb") as f:
            temp_stats = {}
            temp_stats["freq"] = len(pks) / dFF_img.shape[0] * fs
            isi = np.diff(pks) / fs
            temp_stats["isi_mean"] = np.nanmean(isi)
            temp_stats["isi_std"] = np.nanstd(isi)
            temp_stats["n_pks"] = len(pks)
            pickle.dump(temp_stats, f)
    return sta, spike_images


# SEGMENTATION


def identify_hearts(
    img,
    prev_coms=None,
    prev_mask_labels=None,
    fill_missing=True,
    band_bounds=(0.1, 2),
    opening_size=5,
    dilation_size=15,
    method="freq_band",
    corr_threshold=0.9,
    bbox_offset=5,
    **initial_segmentation_args,
):
    """Pick out hearts from widefield experiments using PCA and power content in frequency band"""
    #     print("band_bounds:", band_bounds)
    #     print("band_threshold:", band_threshold)
    #     print("opening_size:", opening_size)
    #     print("dilation_size:", dilation_size)
    #     print("f_s:", f_s)
    mean_img = img.mean(axis=0)
    zeroed_image = img - mean_img

    if method == "freq_band":
        initial_guesses = segment_by_frequency_band(
            img, band_bounds, **initial_segmentation_args
        )
    elif method == "pca_moments":
        initial_guesses = segment_by_pca_moments(img, **initial_segmentation_args)
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
    corr_img = np.zeros((n_rois, mean_img.shape[0], mean_img.shape[1]), dtype=float)
    if corr_threshold == "None":
        new_mask = initial_guesses.astype(bool)
    else:
        corr_threshold = float(corr_threshold)
        for i in range(1, n_rois + 1):
            bbox = bboxes[i - 1]
            r1 = max(bbox[0] - bbox_offset, 0)
            c1 = max(bbox[1] - bbox_offset, 0)
            r2 = min(bbox[2] + bbox_offset, initial_guesses.shape[0])
            c2 = min(bbox[3] + bbox_offset, initial_guesses.shape[1])
            #         print(r1,r2,c1,c2)
            roi_mask = np.zeros_like(initial_guesses, dtype=bool)
            roi_mask[r1:r2, c1:c2] = 1
            initial_trace = extract_mask_trace(zeroed_image, mask=roi_mask)
            roi_traces = zeroed_image[:, roi_mask]
            #         print(roi_traces.shape)
            corrs = np.apply_along_axis(
                lambda x: stats.pearsonr(initial_trace, x)[0], 0, roi_traces
            )
            corrs = corrs.reshape((r2 - r1, c2 - c1))
            corr_img[i - 1, r1:r2, c1:c2] = corrs

        corr_mask = morphology.binary_opening(
            np.max(corr_img > corr_threshold, axis=0),
            selem=morphology.disk(opening_size),
        )
        new_mask = morphology.binary_dilation(
            corr_mask, selem=morphology.disk(dilation_size)
        )
    new_mask_labels = measure.label(new_mask)
    coms = ndimage.center_of_mass(
        new_mask,
        labels=new_mask_labels,
        index=np.arange(1, np.max(new_mask_labels) + 1),
    )
    coms = np.array(coms)

    # print(coms.shape)
    return new_mask_labels, coms


def segment_by_frequency_band(
    img,
    band_bounds,
    f_s=1,
    band_threshold=0.45,
    block_size=375,
    offset=5,
    manual_intensity_mask=None,
):
    """Get segmentation by relative power content within a range of frequencies"""
    mean_img = img.mean(axis=0)
    zeroed_image = img - mean_img
    if manual_intensity_mask is None:
        local_thresh = filters.threshold_local(
            mean_img, block_size=block_size, offset=offset
        )
        # intensity_mask = mean_img > np.percentile(mean_img, intensity_threshold*100)
        intensity_mask = mean_img > local_thresh
    else:
        intensity_mask = manual_intensity_mask

    pixelwise_fft = fft(zeroed_image, axis=0)
    N_samps = img.shape[0]
    fft_freq = fftfreq(N_samps, 1 / f_s)[: N_samps // 2]
    abs_power = np.abs(pixelwise_fft[: N_samps // 2, :, :]) ** 2
    norm_abs_power = abs_power / np.sum(abs_power, axis=0)
    band_power = np.sum(
        norm_abs_power[(fft_freq > band_bounds[0]) & (fft_freq < band_bounds[1]), :, :],
        axis=0,
    )
    smoothed_band_power = filters.median(
        band_power, selem=np.ones((5, 5))
    ) * intensity_mask.astype(int)
    processed_band_power = morphology.binary_opening(
        (smoothed_band_power > band_threshold), selem=np.ones((3, 3))
    )
    return measure.label(processed_band_power)


def segment_by_pca_moments(
    img,
    n_components=40,
    krt_threshold=40,
    skw_threshold=0,
    ev_threshold=0.3,
    comp_mag_pct=95,
):
    """Segment a widefield video with multiple embryos using moments of principal components (heart flickers will be small compared to FOV)"""
    mean_img = img.mean(axis=0)
    zeroed_img = img - mean_img
    rd = zeroed_img.reshape((zeroed_img.shape[0], -1))
    # print(rd.shape)
    n_components = min(n_components, rd.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(rd)
    comp_magnitudes = np.abs(pca.components_)
    valid_components = (
        (stats.kurtosis(comp_magnitudes, axis=1) > krt_threshold)
        & (stats.skew(comp_magnitudes, axis=1) > skw_threshold)
        & (pca.explained_variance_ratio_ > ev_threshold)
    )
    mask = np.zeros_like(mean_img)
    for comp_idx in np.argwhere(valid_components).ravel():
        comp = comp_magnitudes[comp_idx].reshape(img.shape[1], img.shape[2])
        mask += comp > np.percentile(comp, comp_mag_pct)
    mask = mask > 0
    return measure.label(mask)


def get_heart_mask(
    img: npt.NDArray,
    n_components: int = 10,
    krt_thresh: float = 40,
    corr_thresh: float = 0.8,
    min_size=40,
    max_size=150,
    plot: bool = False,
):
    """Extract a binary mask identifying a heart from an image of a
    single embryo.

    Extraction based on high kurtosis principal components and size of segmented regions.

    Args:
        img: A 3D array of shape (n_frames, n_rows, n_cols) containing a video
            of a single embryo.
        n_components: Number of principal components to use for segmentation.
        krt_thresh: Kurtosis threshold for identifying principal components
            corresponding to heart dynamics.
        corr_thresh: Correlation threshold for identifying pixels corresponding
            to the principal components.
        min_size: Minimum size of the initial segmentation.
        max_size: Maximum size of the initial segmentation.
        plot: If True, plot the results of the segmentation.
    Returns:
        A 2D boolean array of shape (n_rows, n_cols) containing the mask.
    """
    # Compute the principal components
    pca = PCA(n_components=n_components)
    datmatrix = np.copy(img).reshape(img.shape[0], -1)
    datmatrix -= np.mean(datmatrix, axis=0)
    pca.fit(datmatrix)
    # Calculate filtering criteria: kurtosis and size of contiguous regions
    krt = stats.kurtosis(pca.components_, axis=1)
    valid_size = np.zeros(krt.shape, dtype=bool)
    valid_krt = krt > krt_thresh
    rough_masks = np.zeros((n_components, img.shape[1], img.shape[2]), dtype=bool)

    for i, comp in enumerate(pca.components_):
        # Filter outlier regions based on size
        pc_img = np.abs(comp.reshape(img.shape[1], img.shape[2]))
        mask = pc_img > 4 * np.std(pc_img)
        mask = morphology.binary_opening(mask, footprint=np.ones((3, 3)))
        mask = morphology.binary_closing(mask, footprint=np.ones((3, 3)))
        labels = measure.label(mask)
        areas = np.bincount(labels.ravel())
        if len(areas) < 2:
            valid_size[i] = False
            continue
        max_area_loc = np.argmax(areas[1:]) + 1
        max_area = areas[max_area_loc]
        valid_size[i] = (max_area > min_size) & (max_area < max_size)
        rough_masks[i] = labels == max_area_loc

    valid_components = valid_krt & valid_size

    if plot:
        visualize.plot_pca_data(
            pca,
            datmatrix,
            img.shape[1:],
            n_components=n_components,
            pc_title=lambda i, comp: f"kurtosis: {krt[i]:.2f}, accepted: {valid_components[i]}",
        )

    n_valid_components = np.sum(valid_components)
    print(n_valid_components, end=", ")
    if n_valid_components == 0:
        return np.zeros((img.shape[1], img.shape[2]), dtype=bool)

    comp_idx = np.argmax(krt[valid_components])
    rough_mask = rough_masks[valid_components][comp_idx]

    test_trace = extract_mask_trace(img, mask=rough_mask)
    # Pixel-wise correlation with the selected principal component
    corrs = np.apply_along_axis(
        lambda x: stats.pearsonr(test_trace, x)[0], 0, datmatrix
    )
    corrs = corrs.reshape(img.shape[1], img.shape[2])
    mask = corrs > corr_thresh
    mask = morphology.binary_opening(mask, footprint=np.ones((3, 3)))
    mask = morphology.binary_closing(mask, footprint=np.ones((3, 3)))
    mask = morphology.binary_dilation(mask, footprint=np.ones((3, 3)))

    return mask


def remove_large_jumps(mask_video, direction="rev", n_ref=10):
    if direction == "rev":
        mask_video = np.flip(mask_video, axis=0)

    init_ref_mask = np.median(mask_video[:n_ref], axis=0).astype(bool)

    mask_jumps_removed = np.zeros_like(mask_video, dtype=bool)
    timepoint_filled = np.zeros(mask_jumps_removed.shape[0], dtype=bool)

    for idx, mask in enumerate(mask_video):
        if idx == 0:
            prev_mask = init_ref_mask
        else:
            prev_mask = mask_jumps_removed[idx - 1]

        if np.sum(mask & prev_mask) == 0:
            mask_jumps_removed[idx] = prev_mask
            timepoint_filled[idx] = True
        else:
            mask_jumps_removed[idx] = mask

    if direction == "rev":
        mask_jumps_removed = np.flip(mask_jumps_removed, axis=0)
    return mask_jumps_removed


def refine_segmentation_pca(img, rois, n_components=10, threshold_percentile=70):
    """Refine manual segmentation to localize the heart using PCA assuming that transients are the largest fluctuations present.
    Returns cropped region images and masks for each unique region in the global image mask.

    """

    def pca_component_select(pca):
        """TBD: a smart way to decide the number of principal components. for now just return 1."""
        n_components = 1
        selected_components = np.zeros_like(pca.components_[0])
        fraction_explained_variance = pca.explained_variance_ / np.sum(
            pca.explained_variance_
        )
        for comp_index in range(n_components):
            selected_components = (
                selected_components
                + pca.components_[comp_index] * fraction_explained_variance[comp_index]
            )
        return selected_components

    region_masks = []
    region_data, region_pixels = extract_all_region_data(img, rois)
    for region_idx in range(len(region_data)):
        rd = region_data[region_idx]
        gc = region_pixels[region_idx]
        rd_bgsub = rd - np.mean(rd, axis=0)
        try:
            pca = PCA(n_components=n_components)
            pca.fit(rd_bgsub)
            selected_components = np.abs(pca_component_select(pca))

            indices_to_keep = selected_components > np.percentile(
                selected_components, threshold_percentile
            )
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
    """Run heart segmentation for widefield experiments on all files in a folder.
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
            curr_labels, curr_coms = identify_hearts(raw, **segmentation_args)
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


def link_frames(
    curr_labels, prev_labels, prev_coms, radius=15, propagate_old_labels=True
):
    """Connect two multi-ROI segmentations adjacent in time"""
    curr_mask = curr_labels > 0
    all_curr_labels = np.unique(curr_labels)[1:]
    all_prev_labels = np.unique(prev_labels)[1:]

    curr_coms = ndimage.center_of_mass(
        curr_mask, labels=curr_labels, index=all_curr_labels
    )
    curr_coms = np.array(curr_coms)
    if len(curr_coms.shape) == 2 and len(prev_coms) > 0:
        curr_coms = curr_coms[:, 0] + 1j * curr_coms[:, 1]

        mindist, mindist_indices = utils.pairwise_mindist(curr_coms, prev_coms)
        link_curr = np.argwhere(mindist < radius).ravel()

        link_prev = set(mindist_indices[link_curr] + 1)

        link_curr = set(link_curr + 1)
    elif len(curr_coms.shape) == 1 or len(prev_coms) == 0:
        link_curr = set([])
        link_prev = set([])
    all_curr_labels_set = set(all_curr_labels)
    all_prev_labels_set = set(all_prev_labels)

    new_labels = np.zeros_like(curr_labels)

    for label in link_curr:
        new_labels[curr_labels == label] = all_prev_labels[mindist_indices[label - 1]]

    if propagate_old_labels:
        unassigned_prev_labels = all_prev_labels_set - link_prev
        for label in unassigned_prev_labels:
            new_labels[prev_labels == label] = label

    unassigned_curr_labels = all_curr_labels_set - link_curr
    new_rois_counter = 0
    try:
        starting_idx = np.max(all_prev_labels) + 1
    except Exception:
        starting_idx = 1
    for label in unassigned_curr_labels:
        if np.all(new_labels[curr_labels == label] == 0):
            new_labels[curr_labels == label] = starting_idx + new_rois_counter
            new_rois_counter += 1
    new_mask = new_labels > 0
    new_coms = ndimage.center_of_mass(
        new_mask, labels=new_labels, index=np.unique(new_labels)[1:]
    )
    new_coms = np.array(new_coms)
    try:
        new_coms = new_coms[:, 0] + 1j * new_coms[:, 1]
    except Exception as e:
        new_coms = []
    return new_labels, new_coms


def link_stack(stack, step=-1, radius=15, propagate_old_labels=True):
    if step < 0:
        curr_t = 1
    else:
        curr_t = stack.shape[0] - 2

    prev_labels = stack[curr_t + step]
    prev_mask = prev_labels > 0

    prev_coms = ndimage.center_of_mass(
        prev_mask, labels=prev_labels, index=np.arange(1, np.max(prev_labels) + 1)
    )
    prev_coms = np.array(prev_coms)
    if len(prev_coms.shape) == 2:
        prev_coms = prev_coms[:, 0] + 1j * prev_coms[:, 1]
    new_labels = [prev_labels]
    while curr_t >= 0 and curr_t < stack.shape[0]:
        curr_labels = stack[curr_t]
        try:
            curr_labels, curr_coms = link_frames(
                curr_labels,
                prev_labels,
                prev_coms,
                radius=radius,
                propagate_old_labels=propagate_old_labels,
            )
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


def filter_by_appearances(linked_vid, unlinked_vid, threshold=1 / 3):
    roi_found = []
    for roi in np.arange(1, np.max(linked_vid) + 1):
        roi_linked = linked_vid == roi
        found_in_unlinked = []
        for i in range(roi_linked.shape[0]):
            detected = unlinked_vid[i][roi_linked[i]]
            found_in_unlinked.append(np.any(detected > 0) * (len(detected) > 0))
        roi_found.append(found_in_unlinked)
    roi_found = np.array(roi_found)
    keep = (
        np.argwhere(np.sum(roi_found, axis=1) > threshold * linked_vid.shape[0]).ravel()
        + 1
    )

    filtered_vid = np.zeros_like(linked_vid)
    for idx, roi in enumerate(keep):
        filtered_vid[linked_vid == roi] = idx + 1

    return filtered_vid


def fill_missing_timepoints(
    vid: npt.NDArray, min_size: int = 100, max_size: int = MAX_INT32
):
    """Detect when an ROI drops out of a sequence of movies. This is when
    it becomes too big or too small. Fill in the missing frames with the ROI
    nearest in time that is not too big or too small.

    Args:
        vid: 3D array of ROIs
        min_size: minimum size of an ROI to be considered valid
        max_size: maximum size of an ROI to be considered valid
    Returns:
        A 3D array of ROIs with missing frames filled in.
    """
    roi_sizes = []
    for roi in range(1, np.max(vid) + 1):
        roi_sizes.append(np.sum(vid == roi, axis=(1, 2)))
    roi_sizes = np.array(roi_sizes).reshape(-1, vid.shape[0])

    valid_roi_tpoints = (roi_sizes >= min_size) & (roi_sizes <= max_size)
    try:
        closest_above_threshold = np.apply_along_axis(
            utils.closest_non_zero, 1, valid_roi_tpoints
        ).squeeze()
    except ValueError:
        return vid

    closest_above_threshold = closest_above_threshold.reshape(np.max(vid), -1)
    filled_vid = np.zeros_like(vid)
    for i in range(1, closest_above_threshold.shape[0] + 1):
        replaced_vals = vid[closest_above_threshold[i - 1], :, :] == i
        filled_vid[replaced_vals] = i
    return filled_vid


def get_regularized_mask(img, rois):
    """Correct noisy segmentation using the fact that the hearts are roughly the same size. Take the mean area and draw a circle with the same area centered at the centroid of each original ROI."""
    mask_labels = np.unique(rois)
    mask_labels = mask_labels[mask_labels != 0]

    rps = measure.regionprops(rois)

    areas = [p["area"] for p in rps]
    centroids = np.array([p["centroid"] for p in rps])
    mean_area = np.mean(areas)
    radius = (mean_area / np.pi) ** 0.5

    area_regularized_mask = np.zeros_like(refined_mask)
    for i in range(len(region_masks)):
        rr, cc = draw.disk(tuple(centroids[i, :]), radius)
        valid_indices = (
            (rr < refined_mask.shape[0])
            & (cc < refined_mask.shape[1])
            & (rr >= 0)
            & (cc >= 0)
        )
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
    guess_props = pd.DataFrame(
        measure.regionprops_table(
            guess_objs, properties=("label", "equivalent_diameter_area")
        )
    )
    median_diameter = np.median(guess_props["equivalent_diameter_area"])
    # print(median_diameter)
    coords = feature.peak_local_max(
        dist,
        footprint=morphology.disk(median_diameter / 2 * 0.8),
        labels=mask,
        min_distance=int(median_diameter * 0.5),
        exclude_border=5,
    )
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
        img, block_size, offset=offset, **threshold_params
    )
    mask = img > threshold
    radius = 7
    mask = utils.pad_func_unpad(
        mask,
        lambda x: morphology.binary_opening(x, footprint=morphology.disk(radius)),
        radius,
        constant_values=0,
    )
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
    sgn = np.ones((direction_vectors.shape[0])) - 2 * (direction_vectors[:, 0] < 0)
    direction_vectors *= sgn[:, np.newaxis]
    #     print(direction_vectors)
    theta = -np.median(np.arctan2(-direction_vectors[:, 1], direction_vectors[:, 0]))
    return theta


def remap_regions(regions, props):
    """
    Renumber regions based on known array formation (centroids may be slightly different but we want to fix them). Column changing fastest.
    """
    centroid_y = props["centroid-0"].to_numpy()
    # print(centroid_y)
    cl = cluster.AffinityPropagation().fit(centroid_y.reshape(-1, 1))
    regularized_centroid_y = np.zeros(props.shape[0])
    for label in range(np.max(cl.labels_) + 1):
        regularized_centroid_y[cl.labels_ == label] = np.mean(
            centroid_y[cl.labels_ == label]
        )
    props["regularized_centroid-0"] = regularized_centroid_y
    props = props.sort_values(by=["regularized_centroid-0", "centroid-1"])
    remapped_regions = map_array(
        regions, props["label"].to_numpy(), np.arange(props.shape[0]) + 1
    )
    return remapped_regions


def split_embryos(
    img,
    block_size=201,
    offset=0,
    extra_split_arrays=[],
    min_obj_size=1000,
    manual_roi_seeds=None,
    manual_bbox_size=None,
    **threshold_params,
):
    """Use assumption that embryos lie on a grid to split a widefield video into videos for individual embryos. This is useful when panning between multiple FOVs that may not necessarily be aligned."""
    mean_img = img.mean(axis=0)
    mask, regions = segment_whole_embryos(mean_img, block_size, offset, min_obj_size)
    # fig1, ax1 = plt.subplots(figsize=(4,4))
    # ax1.imshow(regions)
    if manual_roi_seeds is not None:
        for r in range(1, regions.max() + 1):
            if np.sum(manual_roi_seeds[regions == r]) == 0:
                regions[regions == r] = 0
        regions, _, _ = segmentation.relabel_sequential(regions)
    # Determine rotation of embryos
    props_table = pd.DataFrame(
        measure.regionprops_table(regions, properties=["label", "centroid"])
    )
    centroids = np.array(props_table[["centroid-1", "centroid-0"]])
    theta = calculate_dish_rotation(centroids)
    rotated_im = np.array(
        [
            transform.rotate(
                img[i], theta * 180 / np.pi, resize=False, preserve_range=True
            )
            for i in range(img.shape[0])
        ]
    )
    # Rotate and relabel segmented objects
    rotated_regions = transform.rotate(
        regions, theta * 180 / np.pi, cval=0, resize=False, order=0, preserve_range=True
    )
    rotated_props = pd.DataFrame(
        measure.regionprops_table(
            rotated_regions, properties=["label", "bbox", "centroid"]
        )
    )
    rotated_regions = remap_regions(rotated_regions, rotated_props)
    rotated_props = rotated_props.sort_values(
        by=["regularized_centroid-0", "centroid-1"]
    )
    rotated_props["label"] = np.arange(rotated_props.shape[0]) + 1

    embryo_images = []
    processed_extra_arrays = []
    for i in range(rotated_props.shape[0]):
        row = rotated_props.iloc[i]
        if manual_bbox_size:
            end_y = min(int(row["bbox-0"] + manual_bbox_size[0]), rotated_im.shape[0])
            end_x = min(int(row["bbox-1"] + manual_bbox_size[1]), rotated_im.shape[1])
        else:
            end_y = int(row["bbox-2"])
            end_x = int(row["bbox-3"])
        cropped_im = rotated_im[
            :, int(row["bbox-0"]) : end_y, int(row["bbox-1"]) : end_x
        ]
        if manual_bbox_size:
            pad_y = max(manual_bbox_size[0] - (end_y - int(row["bbox-0"])), 0)
            pad_x = max(manual_bbox_size[1] - (end_x - int(row["bbox-1"])), 0)
            # print(pad_y, pad_x)
            cropped_im = np.pad(cropped_im, np.array([[0, 0], [0, pad_y], [0, pad_x]]))
            # print(cropped_im.shape)

        embryo_images.append(cropped_im)

    for arr in extra_split_arrays:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.imshow(arr)
        processed_extra_arrays.append([])
        if len(arr.shape) < 3:
            arr = np.expand_dims(arr, 0)

        rotated_array = np.array(
            [
                transform.rotate(
                    arr[i],
                    theta * 180 / np.pi,
                    cval=0,
                    preserve_range=True,
                    resize=False,
                )
                for i in range(arr.shape[0])
            ]
        )

        for i in range(rotated_props.shape[0]):
            row = rotated_props.iloc[i]
            processed_extra_arrays[-1].append(
                np.squeeze(
                    rotated_array[
                        :,
                        int(row["bbox-0"]) : int(row["bbox-2"]),
                        int(row["bbox-1"]) : int(row["bbox-3"]),
                    ].astype(arr.dtype)
                )
            )
    return embryo_images, rotated_props, rotated_regions, processed_extra_arrays


def translate_image(img: npt.NDArray, shift: Tuple[float, float]) -> npt.NDArray:
    """Translate an image by a shift (x,y)
    Args:
        img: Image to translate
        shift: Tuple of (x,y) shift
    Returns:
        Translated image
    """
    u, v = shift
    nr, nc = img.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    return transform.warp(
        img, np.array([row_coords - v, col_coords - u]), mode="constant", cval=np.nan
    )


def match_snap_to_data(
    img: npt.NDArray, ref: npt.NDArray, scale_factor: float = 4
) -> npt.NDArray:
    """Match a snap of arbitrary size to an equally sized or smaller reference data set, assuming
    both ROIs are centered in camera coordinates

    Args:
        img: Image to match
        ref: Reference image to match to
        scale_factor: Factor to downscale the image by before matching
    Returns:
        Cropped image

    """
    downscaled = transform.downscale_local_mean(img, (scale_factor, scale_factor))
    cropped, _ = crop_min_shape(downscaled, ref)
    return cropped


def crop_min_shape(
    im1: npt.NDArray, im2: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Crops two images to the minimum shape of the two, assuming the images are centered
    in the camera coordinates

    Args:
        im1: First image to crop
        im2: Second image to crop
    Returns:
        Cropped images
    """
    min_shape = np.min([im1.shape[:2], im2.shape[:2]], axis=0)
    diff_shape_im1 = np.array(im1.shape[:2]) - min_shape
    diff_shape_im2 = np.array(im2.shape[:2]) - min_shape
    cropped_im1 = im1[
        diff_shape_im1[0] // 2 : diff_shape_im1[0] // 2 + min_shape[0],
        diff_shape_im1[1] // 2 : diff_shape_im1[1] // 2 + min_shape[1],
    ]
    cropped_im2 = im2[
        diff_shape_im2[0] // 2 : diff_shape_im2[0] // 2 + min_shape[0],
        diff_shape_im2[1] // 2 : diff_shape_im2[1] // 2 + min_shape[1],
    ]
    return cropped_im1, cropped_im2

def extract_roi(img, x0, width, y0,  height):
    """Extracts a region of interest from an image
    Args:
        img: Image to extract from
        y0: Top left y coordinate
        x0: Top left x coordinate
        height: Height of ROI
        width: Width of ROI
    Returns:
        Cropped image
    """
    return img[y0 : y0 + height, x0 : x0 + width]