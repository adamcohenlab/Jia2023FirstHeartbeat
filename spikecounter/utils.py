"""
Utility functions for spikecounter package

"""
from pathlib import Path
import warnings
import importlib
import os
from os import PathLike
from typing import Union, List, Tuple, Dict, Any, Callable

import numpy as np
from numpy import typing as npt
from skimage import transform
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
from ipywidgets import interact
from scipy import interpolate, signal
import re
import pandas as pd
import mat73



def custom_newton_lsq(
    x: float,
    y: float,
    gx: Callable[[npt.ArrayLike], npt.NDArray],
    dgx: Callable[[npt.ArrayLike], npt.NDArray],
    bounds: Tuple[float, float] = (-np.inf, np.inf),
) -> float:
    """solve for the x that gives a particular value of a function g, i.e. (g(x) - y)**2 = 0

    Args:
        x: initial guess
        y: known value of g(x)
        gx: function g(x)
        dgx: derivative of g(x)
        bounds: bounds for the solution
    Returns:
        x1: Updated guess for the solution

    """
    y_est = float(gx(x))
    dy_est = float(dgx(x))
    fx = y_est**2 - 2 * y * y_est + y**2
    dfx = 2 * dy_est * y_est - 2 * y * dy_est
    x1 = x - fx / dfx
    x1 = max(bounds[0], x1)
    x1 = min(bounds[1], x1)
    return x1


def round_rel_deviation(a, factor=100):
    arr = np.atleast_1d(a)
    try:
        n_decimals = -int(np.floor(np.min(np.log10(np.abs(arr[arr != 0]) / factor))))
    except ValueError:
        n_decimals = 1
    rounded = np.squeeze(np.round(arr, n_decimals))
    #     print(arr, n_decimals, rounded)
    return rounded


def pad_func_unpad(arr, func, pad_width, **pad_params):
    """Control function behavior around edge of arrays (2D)"""
    padded_array = np.pad(arr, pad_width, **pad_params)
    output = func(padded_array)
    unpadded = output[pad_width:-pad_width, pad_width:-pad_width]
    return unpadded


def make_iterable(x) -> Any:
    if isinstance(x, str):
        yield x
    else:
        try:
            yield from x
        except TypeError:
            yield x


def reload_libraries(libraries):
    """Reload libraries from a list of strings
    Args:
        libraries (list): list of strings of libraries to reload
    Returns:
        None
    """
    for lib in make_iterable(libraries):
        importlib.reload(lib)

def interpolate_invalid_values(
    arr: npt.NDArray, mask: npt.NDArray[np.bool_], kind: str = "previous"
):
    """Interpolate invalid values along the first axis of an N-D array

    Inputs:
        arr: array to be interpolated
        mask: 1D array of mask data (pixels x pixels). True values indicate invalid values.
    Returns:
        Array with invalid values interpolated.
    """
    trace_length = arr.shape[0]
    xs = np.arange(trace_length, dtype=int)[mask]
    invalid_filled = np.copy(arr)
    if kind == "previous":
        invalid_filled[xs] = arr[
            closest_non_zero(~mask, direction="left").squeeze()[xs]
        ]
    elif kind == "linear":
        closest_left = closest_non_zero(~mask, direction="left").squeeze()[xs]
        closest_right = closest_non_zero(~mask, direction="right").squeeze()[xs]
        weight = (xs - closest_left) / (closest_right - closest_left)
        invalid_filled[xs] = (
            arr[closest_left] * (1 - weight)[:, None, None]
            + arr[closest_right] * weight[:, None, None]
        )
    return invalid_filled


def shiftkern(kernel, a, b, c, dt):
    """From Hochbaum and Cohen 2012"""
    k = a * kernel + b
    t0 = np.argmax(kernel)
    L = len(kernel)
    x1 = np.arange(L) - t0
    x2 = (x1 - dt) * c
    interpf = interpolate.interp1d(
        x1, k, fill_value=np.percentile(k, 10), bounds_error=False
    )
    return interpf(x2)


def extract_experiment_name(input_path):
    folder_names = input_path.split("/")
    if folder_names[-1] == "":
        expt_name = folder_names[-2]
    else:
        expt_name = folder_names[-1]
    expt_name = expt_name.split(".tif")[0]
    return expt_name


def load_experiment_metadata(
    root_dir: Union[str, PathLike],
    expt_name: str,
) -> Union[Dict[str, Any], None]:
    """Load and interpret metadata file from experiment folder

    Args:
        root_dir (str): path to experiment folder
        expt_name (str): name of experiment folder

    Returns:
        expt_data (Union[Dict[str, Any], None]): dictionary of experiment metadata

    """
    metadata_path = Path(root_dir, expt_name, "output_data_py.mat")
    if metadata_path.exists():
        try:
            expt_data = mat73.loadmat(
                os.path.join(root_dir, expt_name, "output_data_py.mat")
            )["dd_compat_py"]
        except FileNotFoundError as err:
            warnings.warn(str(err))
            expt_data = None
    else:
        metadata_path = Path(root_dir, expt_name, "experimental_parameters.txt")
        try:
            expt_data = {"camera": {"roi": [0, 0, 0, 0]}}
            with metadata_path.open() as f:
                expt_data["camera"]["roi"][1] = int(
                    re.search("\d+", f.readline()).group(0)
                )
                expt_data["camera"]["roi"][3] = int(
                    re.search("\d+", f.readline()).group(0)
                )
        except FileNotFoundError as err:
            warnings.warn(str(err))
            expt_data = None
    return expt_data


def process_experiment_metadata(
    expt_metadata: pd.DataFrame,
    regexp_dict: Union[Dict, None] = None,
    dtypes: Union[Dict, None] = None,
):
    """Extract data from filenames in basic metadata table.

    Args:
        expt_metadata: basic metadata table
        regexp_dict: optional dictionary of regular expressions to extract data from filenames
        dtypes: optional dictionary of data types for extracted data
    Returns:
        A metadata table with information parsed from filenames.

    """
    new_df = expt_metadata.sort_values("start_time").reset_index()
    if "index" in new_df:
        del new_df["index"]
    start_times = [datetime.strptime(t, "%H:%M:%S") for t in list(new_df["start_time"])]
    offsets = [s - start_times[0] for s in start_times]
    offsets = [o.seconds for o in offsets]
    new_df["offset"] = offsets
    if regexp_dict:
        for key, value in regexp_dict.items():
            parser_results = [re.search(value, f) for f in list(new_df["file_name"])]
            matches = []
            for res in parser_results:
                if res:
                    matches.append(res.group(0))
                else:
                    matches.append("None")
            new_df[key] = matches
            if dtypes and key in dtypes:
                new_df[key] = new_df[key].astype(dtypes[key])
    return new_df


def match_experiments_to_snaps(expt_data, snap_data):
    # print("test")
    snap_files = []
    snap_data_by_embryo = snap_data.set_index("embryo")
    for i in range(expt_data.shape[0]):
        try:
            embryo_snap_data = snap_data_by_embryo.loc[[expt_data.iloc[i]["embryo"]]]
            # print(embryo_snap_data)
            start_time = datetime.strptime(expt_data.iloc[i]["start_time"], "%H:%M:%S")
            # print(embryo_snap_data[["start_time"]])
            snap_times = [
                datetime.strptime(t, "%H:%M:%S")
                for t in list(embryo_snap_data["start_time"].values.ravel())
            ]
            diffs = [abs(start_time - t).seconds for t in snap_times]
            snap_idx = np.argmin(diffs)
            snap_files.append(embryo_snap_data["file_name"].iloc[snap_idx])
        except KeyError as e:
            # print(e)
            snap_files.append(None)
    return pd.concat(
        [expt_data, pd.DataFrame(snap_files, columns=["snap_file"])], axis=1
    )
    # expt_data["snap_file"] = snap_files


def generate_file_list(input_path):
    if os.path.isdir(input_path):
        raw_files = sorted(
            [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if os.path.splitext(f)[1] == ".tif"
            ]
        )
        files = []
        idx = 0
        while idx < len(raw_files):
            if "block" in raw_files[idx]:
                filename = os.path.splitext(os.path.basename(raw_files[idx]))[0].split(
                    "_block"
                )[0]
                block = [raw_files[idx]]
                part_of_block = True
                while part_of_block and idx < len(raw_files) - 1:
                    idx += 1
                    if filename in raw_files[idx]:
                        block += [raw_files[idx]]
                    else:
                        part_of_block = False
                        idx -= 1
                files += [block]
            else:
                files += [raw_files[idx]]
            idx += 1

    else:
        files = [input_path]
    return files


def make_output_folder(input_path=None, output_path=None, make_folder_from_file=False):
    if output_path is None:
        if os.path.isdir(input_path):
            output_folder = input_path
        else:
            if make_folder_from_file:
                output_folder = os.path.join(
                    os.path.dirname(input_path),
                    os.path.splitext(os.path.basename(input_path))[0],
                )
            else:
                output_folder = os.path.dirname(input_path)
    else:
        output_folder = output_path

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.isdir(output_folder):
        raise Exception("Generated output path is not a folder")

    return output_folder


def write_subfolders(output_folder, subfolders):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for subfolder in subfolders:
        try:
            os.mkdir(os.path.join(output_folder, subfolder))
        except Exception:
            pass


def standardize_n_dims(img, missing_dims=None):
    if missing_dims is not None:
        img = np.expand_dims(img, tuple(missing_dims))
    if len(img.shape) == 6:
        return img[:, 0, :, :, :, :]
    n_axes_to_add = 5 - len(img.shape)
    if n_axes_to_add < 1:
        return img
    else:
        print(img.shape)
        print(np.arange(n_axes_to_add))
        return np.expand_dims(img, tuple(list(np.arange(n_axes_to_add))))


def img_to_8bit(img):
    img_8bit = img / np.max(img) * 255
    return img_8bit.astype(np.uint8)


def project_y(img, z_to_x_ratio=1):
    print(img.shape)
    max_proj_y = np.expand_dims(img.max(axis=3), 3)
    max_proj_y = np.swapaxes(max_proj_y, 1, 3)
    # print(np.max(max_proj_y[:,:,0,:,:]))
    # print(np.max(max_proj_y[:,:,1,:,:]))
    # quit()
    max_proj_y_rescaled = np.zeros(
        (
            max_proj_y.shape[0],
            max_proj_y.shape[1],
            max_proj_y.shape[2],
            int(np.round(max_proj_y.shape[3] * z_to_x_ratio)),
            max_proj_y.shape[4],
        ),
        dtype=max_proj_y.dtype,
    )
    for t in range(img.shape[0]):
        for c in range(img.shape[2]):
            max_proj_y_rescaled[t, 0, c, :, :] = transform.resize(
                max_proj_y[t, 0, c, :, :],
                (
                    int(np.round(max_proj_y.shape[3] * z_to_x_ratio)),
                    max_proj_y.shape[4],
                ),
                preserve_range=True,
                order=3,
            )
    max_proj_y_rescaled = np.flip(max_proj_y_rescaled, axis=3)
    return max_proj_y_rescaled


def project_x(img, z_to_y_ratio=1):
    print(img.shape)
    max_proj_x = np.expand_dims(img.max(axis=4), 4)
    max_proj_x = np.swapaxes(max_proj_x, 1, 4)
    # print(np.max(max_proj_x[:,:,0,:,:]))
    # print(np.max(max_proj_x[:,:,1,:,:]))
    # quit()
    max_proj_x_rescaled = np.zeros(
        (
            max_proj_x.shape[0],
            max_proj_x.shape[1],
            max_proj_x.shape[2],
            max_proj_x.shape[3],
            int(np.round(max_proj_x.shape[4] * z_to_y_ratio)),
        ),
        dtype=max_proj_x.dtype,
    )
    for t in range(img.shape[0]):
        for c in range(img.shape[2]):
            max_proj_x_rescaled[t, 0, c, :, :] = transform.resize(
                max_proj_x[t, 0, c, :, :],
                (
                    max_proj_x.shape[3],
                    (int(np.round(max_proj_x.shape[4] * z_to_y_ratio))),
                ),
                preserve_range=True,
                order=3,
            )
    return max_proj_x_rescaled


def max_entropy(self, raw_img):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    data, _ = exposure.histogram(raw_img, normalize=True)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[: it + 1]
        hist_range = (
            hist_range[hist_range != 0] / cdf[it]
        )  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1 :]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return threshold


def datestring_to_epoch(s, fmt="%Y-%m-%dT%H:%M:%S"):
    return datetime.strptime(s, fmt).timestamp()


def transferjob(sourcedir, targetdir):
    ## Adapted from Daniel Scott Eaton Trenchripper (https://github.com/DanielScottEaton/TrenchRipper)
    mkdircmd = "mkdir -p " + targetdir
    rsynccmd = "rsync -r " + sourcedir + "/ " + targetdir
    wrapcmd = mkdircmd + " && " + rsynccmd
    cmd = 'sbatch -p transfer -t 0-12:00 --wrap="' + wrapcmd + '"'
    os.system(cmd)


def display_zstack(stack, z=0, c="all", markers=[], pct_cutoffs=[5, 95], cmap=None):
    st = stack.copy()
    if len(stack.shape) < 4:
        st = st[:, :, :, np.newaxis]
    min_value = np.percentile(st[~np.isnan(st)], pct_cutoffs[0])
    max_value = np.percentile(st[~np.isnan(st)], pct_cutoffs[1])
    if cmap is None:
        cmap = plt.rcParams["image.cmap"]

    def view_image(z, c):
        if c == "all":
            img = st[z, :, :, :]
        else:
            img = st[z, :, :, int(c)]
        q = plt.imshow(
            img, interpolation="nearest", vmin=min_value, vmax=max_value, cmap=cmap
        )
        plt.colorbar(q)
        if len(markers) > 0:
            for marker in markers:
                plt.plot(marker[0], marker[1], "rx")
        plt.title("Z: %d C: %s" % (z, str(c)))
        plt.show()

    interact(
        view_image, z=(0, st.shape[0] - 1), c=["all"] + list(np.arange(st.shape[-1]))
    )


def convert_to_iterable(x):
    try:
        iterator = iter(x)
    except TypeError:
        iterator = iter([x])
    return iterator


def traces_to_dict(matdata):
    rising_edges = np.argwhere(np.diff(matdata["frame_counter"]) == 1).ravel()
    if isinstance(matdata["task_traces"], dict):
        trace_types = [matdata["task_traces"]]
    else:
        trace_types = matdata["task_traces"]
    dt_dict = {}
    for trace_type in trace_types:
        traces = trace_type["traces"]
        if isinstance(traces["name"], str):
            dt_dict[traces["name"]] = np.zeros_like(rising_edges, dtype=float)

            for j in range(len(rising_edges)):
                if j == 0:
                    dt_dict[traces["name"]][j] = np.max(
                        traces["values"][: rising_edges[j]]
                    )
                else:
                    dt_dict[traces["name"]][j] = np.max(
                        traces["values"][rising_edges[j - 1] : rising_edges[j]]
                    )
        else:
            for i in range(len(traces["name"])):
                dt_dict[traces["name"][i]] = np.zeros_like(rising_edges, dtype=float)
                for j in range(len(rising_edges) - 1):
                    if j == 0:
                        dt_dict[traces["name"][i]][j] = np.max(
                            traces["values"][i][: rising_edges[j]]
                        )
                    else:
                        dt_dict[traces["name"][i]][j] = np.max(
                            traces["values"][i][rising_edges[j - 1] : rising_edges[j]]
                        )
    t = rising_edges / matdata["clock_rate"]
    return dt_dict, t


def combine_jagged_arrays(arrays, justify="left"):
    """Combine traces of different lengths into a 2D array"""
    rows = len(arrays)
    individual_lengths = [len(arr) for arr in arrays]
    cols = np.max(individual_lengths)
    combined = np.ones((rows, cols)) * np.nan
    if justify == "left":
        for i in range(len(arrays)):
            combined[i, : individual_lengths[i]] = arrays[i]
    elif justify == "right":
        for i in range(len(arrays)):
            combined[i, -individual_lengths[i] :] = arrays[i]
    return combined


def align_traces(unaligned_traces, all_index_offsets):
    """Align a list of (blocks of) unaligned traces of varying sizes and known index offsets.
    Return the aligned traces and the global time coordinate.
    """
    n_traces = [ut.shape[0] for ut in unaligned_traces]
    total_traces = np.sum(n_traces)
    max_length = np.max([ut.shape[1] for ut in unaligned_traces])
    max_offset = np.max([np.max(ao) for ao in all_index_offsets])
    n_timepoints = max_length + max_offset

    aligned_traces = np.nan * np.ones((total_traces, n_timepoints))
    curr_row = 0
    for i in range(len(n_traces)):
        ut = unaligned_traces[i]
        ao = all_index_offsets[i]
        for j in range(ut.shape[0]):
            start_idx = aligned_traces.shape[1] - max_length - ao[j]
            aligned_traces[curr_row, start_idx : start_idx + ut.shape[1]] = ut[j, :]
            curr_row += 1
    global_time = np.arange(aligned_traces.shape[1]) - max_offset
    return aligned_traces, global_time


def div(x):
    """Calculate divergence of a N-D vector field in Cartesian coordinates"""
    diag_derivs = []
    for i in range(x.shape[0]):
        dxi = np.gradient(x[i])
        diag_derivs.append(dxi[i])
    divergence = np.sum(np.array(diag_derivs), axis=0)

    return divergence


def closest_non_zero(arr: npt.NDArray, direction="both"):
    """Find the index of the closest non-zero element in an array, in a given direction."""
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        raise ValueError("Array must be 1D")
    if arr.dtype == bool:
        nonzeros = np.argwhere(arr).ravel()
    else:
        nonzeros = np.argwhere(arr != 0).ravel()
    diffs = np.subtract.outer(np.arange(len(arr), dtype=int), nonzeros).reshape(
        (len(arr), -1)
    )

    if direction == "both":
        distances = np.abs(diffs)
    elif direction == "left":
        distances = diffs
        distances = np.ma.masked_array(distances, distances < 0)
    elif direction == "right":
        distances = -diffs
        distances = np.ma.masked_array(distances, distances < 0)
    else:
        distances = np.zeros_like(diffs)

    min_indices = distances.argmin(axis=1)
    return nonzeros[min_indices]


def pairwise_dist(x, y):
    """calculate euclidean distances between lists of vectors x and y, where each row is a measurement"""
    if len(x.shape) > 1:
        pd = (
            np.sum(
                np.array(
                    [np.subtract.outer(x[:, i], y[:, i]) for i in range(x.shape[1])]
                )
                ** 2,
                axis=0,
            )
            ** 0.5
        )
    else:
        # Assume x and y are representation of 2D vectors as complex values
        pd = np.abs(np.subtract.outer(x, y))
    return pd


def pairwise_mindist(x, y):
    """Calculate minimum distance between each point in the list x and each point in the list y"""
    # Get pairwise distances between vectors in x and y
    pd = pairwise_dist(x, y)

    # Check if x and y are identical, i.e. find the minimum distance to the point that is not itself
    if np.all(np.diag(pd) == 0):
        pd += np.diag(np.inf * np.ones(pd.shape[0]))

    # Get minima
    mindist_indices = np.argmin(pd, axis=1)
    mindist = pd[np.arange(x.shape[0]), mindist_indices]

    return mindist, mindist_indices


def space_average_over_time(timeseries, mask=None):
    if mask is not None:
        time_mask = np.zeros_like(timeseries)
        print(time_mask.shape)
        time_mask[0, :, :] = mask
        for t in range(1, time_mask.shape[0]):
            mask[mask != 0] += 1
            time_mask[t, :, :] = mask
        print(time_mask.max())
    return np.array(
        nd.mean(
            timeseries, labels=time_mask, index=np.arange(1, time_mask.shape[0] + 1)
        )
    )


def subsampled_autocorrelation(
    x, window_length, start_idx=0, subtract_mean=True, normalize=True, n_windows=1
):
    """Wrapper function for scipy.signal.autocorrelation that is more human-interpretable"""
    dc_offset_ref = 0
    dc_offset_lag = 0

    ref = x[start_idx : start_idx + window_length]
    lag = x[start_idx : start_idx + (n_windows + 1) * window_length]

    if subtract_mean:
        dc_offset_ref = np.nanmean(ref)
        dc_offset_lag = np.nanmean(ref)

    corr = signal.correlate(ref - dc_offset_ref, lag - dc_offset_lag, mode="valid")
    corr = np.flip(corr)
    if normalize:
        corr /= np.max(corr)
    return corr
