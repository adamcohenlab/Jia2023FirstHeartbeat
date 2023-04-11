""" Type annotations for images.py """
# pylint: disable=unused-argument

from typing import Tuple, Optional, Dict, List, Any, Iterable, Union, TypeVar, Collection
from scipy import interpolate
from typing import overload
from os import PathLike
import numpy.typing as npt
import numpy as np

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
):