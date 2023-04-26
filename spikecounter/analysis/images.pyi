""" Type annotations for images.py """
# pylint: disable=unused-argument

from typing import Tuple, Optional, Dict, List, Any, Iterable, Union, TypeVar, Collection
from scipy import interpolate
from typing import overload
from os import PathLike
import numpy.typing as npt
import numpy as np
from matplotlib import axes

NPGeneric = TypeVar("NPGeneric", bound=np.generic)

def regress_video(
    img: npt.NDArray, trace_array: npt.NDArray, regress_dc: bool = True
) -> npt.NDArray: ...

@overload
def load_image(
    rootdir: Union[str, PathLike[Any]],
    expt_name: str,
    subfolder: str = ...,
    raw: bool = ...,
    cam_indices: int = ...,
    expt_metadata: Optional[Dict[str, Any]] = ...,
) -> Tuple[npt.NDArray[np.generic], Optional[Dict[str, Any]]]: ...
@overload
def load_image(
    rootdir: Union[str, PathLike[Any]],
    expt_name: str,
    subfolder: str = ...,
    raw: bool = ...,
    cam_indices: Iterable[int] = ...,
    expt_metadata: Optional[Dict[str, Any]] = ...,
) -> Tuple[
    Union[npt.NDArray[np.generic], List[npt.NDArray[np.generic]]],
    Optional[Dict[str, Any]],
]: ...
@overload
def load_image(
    rootdir: Union[str, PathLike[Any]],
    expt_name: str,
    subfolder: str = ...,
    raw: bool = ...,
    cam_indices: None = ...,
    expt_metadata: Optional[Dict[str, Any]] = ...,
) -> Tuple[
    Union[npt.NDArray[np.generic], List[npt.NDArray[np.generic]]],
    Optional[Dict[str, Any]],
]: ...


def downsample_video(
    raw_img: npt.NDArray,
    downsample_factor: int,
    aa: Union[str, None] = "gaussian",
) -> npt.NDArray: ...

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
]: ...

def spline_timing(
    img: npt.NDArray, s: float = 0.1, n_knots: int = 4, upsample_rate: float = 1
): ...

def extract_background_traces(
    img: npt.NDArray, mode: Union[str, Collection[str]] = "all"
) -> npt.NDArray: ...

def get_image_dFF(
    img: npt.NDArray[Union[np.floating, np.integer]],
    baseline_percentile: float = 10,
    t_range: Tuple[int, int] = (0, -1),
    invert: bool = False,
):...

def extract_cropped_region_image(
    intensity: npt.NDArray, global_coords: Union[npt.NDArray, Tuple[int, int]]
) -> npt.NDArray:...

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
]: ...

def extract_mask_trace(
    img: npt.NDArray, mask: Optional[npt.NDArray[np.bool_]] = None
) -> npt.NDArray: ...

def spike_triggered_average_video(
    img: npt.NDArray[Union[np.floating, np.integer]],
    peak_indices: npt.NDArray[Union[np.integer, np.bool_]],
    sta_bounds: Tuple[int, int],
    include_mask: Optional[npt.NDArray[np.bool_]] = None,
    normalize_height: bool = False,
    full_output: bool = False,
) -> Tuple[npt.NDArray[np.floating], Optional[npt.NDArray[np.floating]]]: ...