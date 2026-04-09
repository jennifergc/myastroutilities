"""
myastroutilities.pv.extract
============================
Position-velocity diagram extraction and visualization.

Problem solved
--------------
Three scripts (``pvutils.py``, ``pv-gen.py``, ``pv_slice_wcs.py``) contained
near-duplicate implementations of the PV extraction loop. Each script added
minor variations while sharing the same core ``map_coordinates`` call.
This module provides a single, canonical implementation that covers moment-0
computation, PV extraction, FITS output, and two-panel figure generation.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.ndimage import map_coordinates

from myastroutilities.structures import UC1, SpectralWindow
from myastroutilities.cube.io import load_spectral_cube, get_image_metadata
from myastroutilities.cube.moments import run_bettermoments
from myastroutilities.pv.geometry import slice_pixel_coords


# ─────────────────────────────────────────────
#   Moment 0 helper
# ─────────────────────────────────────────────

def compute_moment0(
    fits_path: str,
    prefix: str,
    window: SpectralWindow,
    kernel: int = 50,
    mask: bool = True,
    mask_mult: float = 3.0,
) -> str:
    """
    Computes the moment-0 map for a given spectral window.

    Delegates to :func:`~myastroutilities.cube.moments.run_bettermoments`
    and returns the path of the generated FITS file.

    Parameters
    ----------
    fits_path : str
        Path to the input FITS spectral cube.
    prefix : str
        Base name for the output FITS file.
    window : SpectralWindow
        Channel range to integrate.
    kernel : int, optional
        Number of channels used to estimate the RMS. Defaults to 50.
    mask : bool, optional
        If ``True``, applies an RMS threshold mask. Defaults to ``True``.
    mask_mult : float, optional
        Mask threshold in units of RMS. Defaults to 3.0.

    Returns
    -------
    str
        Absolute path to the moment-0 FITS file.
    """
    run_bettermoments(
        fits_path, prefix,
        window.ch_min, window.ch_max,
        kernel, momento="0",
        mask=mask, mask_mult=mask_mult,
    )
    out_dir = os.path.dirname(fits_path)
    return os.path.join(out_dir, f"{prefix}_momento0_BM.fits")


# ─────────────────────────────────────────────
#   PV extraction
# ─────────────────────────────────────────────

def _extract_pv_array(
    cube,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    window: SpectralWindow,
    interpolation_order: int = 1,
) -> np.ndarray:
    """
    Extracts the PV array from the cube along a spatial path.

    Iterates channel-by-channel and uses ``scipy.ndimage.map_coordinates``
    for sub-pixel interpolation. This pattern was duplicated 10+ times
    across notebooks and scripts.

    Parameters
    ----------
    cube : SpectralCube
        Full spectral cube (not sliced). Accessed channel by channel.
    x_pix, y_pix : ndarray, shape (N,)
        Spatial pixel coordinates along the slice.
    window : SpectralWindow
        Channel range to extract.
    interpolation_order : int, optional
        Order for ``map_coordinates``. Defaults to 1 (bilinear).

    Returns
    -------
    pv : ndarray, shape (N, n_channels)
        PV array: axis 0 = spatial offset, axis 1 = spectral channel.
    """
    n_pos = len(x_pix)
    pv = np.zeros((n_pos, window.n_channels))
    for j, ch in enumerate(window):
        img = cube[ch].value.squeeze()
        pv[:, j] = map_coordinates(img, [y_pix, x_pix],
                                   order=interpolation_order, mode="nearest")
    return pv


def extract_pv(
    cube_path: str,
    mom0_path: str,
    window: SpectralWindow,
    angle: float,
    length_pix: int,
    rms: float,
    source: Optional[SkyCoord] = None,
    ra: Optional[str] = None,
    dec: Optional[str] = None,
    sysvel: Optional[float] = None,
    pv_fits: Optional[str] = None,
    png_out: Optional[str] = None,
    contour_levels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts a PV diagram and plots it alongside the moment-0 map.

    Accepts a source position as either a :class:`~astropy.coordinates.SkyCoord`
    object or as separate ``ra``/``dec`` strings. Falls back to :data:`UC1`
    when no source is provided.

    Parameters
    ----------
    cube_path : str
        Path to the input FITS spectral cube.
    mom0_path : str
        Path to the moment-0 FITS image (used for the spatial panel).
    window : SpectralWindow
        Channel range that defines the velocity axis.
    angle : float
        Position angle of the slice in degrees (measured from north toward east).
    length_pix : int
        Number of sample points along the spatial slice.
    rms : float
        RMS noise level in Jy/beam, used to set contour levels.
    source : SkyCoord, optional
        Sky position of the slice center. Overrides ``ra``/``dec``.
    ra : str, optional
        Right ascension string (e.g., ``'18h20m24.821s'``). Used when
        ``source`` is ``None``.
    dec : str, optional
        Declination string (e.g., ``'-16d11m35.08s'``). Used when
        ``source`` is ``None``.
    sysvel : float, optional
        Systemic velocity in km/s. Reserved for future use.
    pv_fits : str, optional
        If provided, the PV array is saved as a FITS file with WCS headers.
    png_out : str, optional
        If provided, the figure is saved to this path at 600 dpi.
    contour_levels : ndarray, optional
        Contour level multipliers in units of ``rms``. Defaults to
        ``[-1, 2, 20, 50, 100, 300, 500]``.

    Returns
    -------
    pv : ndarray, shape (length_pix, window.n_channels)
        Extracted PV array.

    Examples
    --------
    >>> window = SpectralWindow(930, 1020)
    >>> pv = extract_pv("cube.fits", "moment0.fits", window,
    ...                 angle=126.0, length_pix=200, rms=9.8e-3)
    """
    # --- Resolve source position ---
    if source is None:
        if ra is not None and dec is not None:
            source = SkyCoord(ra=ra, dec=dec, frame="icrs")
        else:
            source = UC1.coord

    if contour_levels is None:
        contour_levels = np.array([-1, 2, 20, 50, 100, 300, 500])

    # --- Load moment-0 image ---
    with fits.open(mom0_path) as hdul:
        moment0_img = hdul[0].data.squeeze()

    # --- Load full cube and subcube ---
    cube = load_spectral_cube(cube_path)
    nchan = cube.shape[0]
    if window.ch_max >= nchan:
        raise ValueError(
            f"Cube has only {nchan} channels (0..{nchan-1}); "
            f"ch_max={window.ch_max} is out of range."
        )
    subcube = cube[window.slice]
    vel_axis = subcube.spectral_axis.to_value("km/s")

    # --- Spatial metadata ---
    meta = get_image_metadata(cube_path)
    pixscale = meta.pixel_scale_arcsec
    x0, y0 = meta.wcs.world_to_pixel(source)

    # --- Slice geometry ---
    angle_rad = np.deg2rad(angle)
    x_pix, y_pix, offsets = slice_pixel_coords(x0, y0, angle_rad, length_pix)
    offset_arc = offsets * pixscale

    # --- Extract PV ---
    pv = _extract_pv_array(cube, x_pix, y_pix, window)

    # --- Save FITS ---
    if pv_fits:
        dv = float(np.mean(np.diff(vel_axis)))
        hdr = fits.Header()
        hdr["WCSAXES"] = 2
        hdr["NAXIS1"] = window.n_channels
        hdr["NAXIS2"] = length_pix
        hdr["CTYPE1"] = "VELO-LSR"
        hdr["CUNIT1"] = "km/s"
        hdr["CRPIX1"] = 1
        hdr["CRVAL1"] = float(vel_axis.max())
        hdr["CDELT1"] = -abs(dv)
        hdr["CTYPE2"] = "OFFSET"
        hdr["CUNIT2"] = "arcsec"
        hdr["CRPIX2"] = 1
        hdr["CRVAL2"] = float(offset_arc.min())
        hdr["CDELT2"] = float(offset_arc[1] - offset_arc[0])
        hdr["BUNIT"] = "Jy/beam"
        hdr["SLICEPA"] = angle
        hdr["CHMIN"] = window.ch_min
        hdr["CHMAX"] = window.ch_max
        fits.PrimaryHDU(pv.astype(np.float32), header=hdr).writeto(
            pv_fits, overwrite=True
        )

    # --- Two-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(top=0.80, bottom=0.05, left=0.08,
                        right=0.95, hspace=0.3, wspace=0.3)

    ny, nx = moment0_img.shape
    extent_map = [
        (0 - x0) * pixscale, (nx - 1 - x0) * pixscale,
        (0 - y0) * pixscale, (ny - 1 - y0) * pixscale,
    ]

    # Left panel: moment 0
    im_map = axes[0].imshow(moment0_img, origin="lower",
                            cmap="inferno", extent=extent_map)
    axes[0].plot(
        offsets * np.cos(angle_rad) * pixscale,
        offsets * np.sin(angle_rad) * pixscale,
        color="white", lw=2,
    )
    axes[0].scatter(0, 0, marker="*", color="yellow",
                    s=120, edgecolor="black", zorder=5)
    axes[0].set_xlabel("ΔRA (arcsec)")
    axes[0].set_ylabel("ΔDec (arcsec)")

    # Right panel: PV diagram
    im_pv = axes[1].imshow(
        pv, origin="lower", aspect="auto",
        extent=[vel_axis.max(), vel_axis.min(),
                offset_arc.min(), offset_arc.max()],
        cmap="jet",
    )
    axes[1].axhline(0, color="white", linestyle="--", lw=1.5)
    X, Y = np.meshgrid(vel_axis, offset_arc)
    axes[1].contour(X, Y, pv, levels=contour_levels * rms,
                    colors="black", linewidths=1, alpha=0.8)
    axes[1].set_xlabel("Velocity (km/s)")
    axes[1].set_ylabel("Offset (arcsec)")

    # Colorbars
    cax_map = fig.add_axes([0.08, 0.84, 0.445, 0.01])
    cbar_map = fig.colorbar(im_map, cax=cax_map, orientation="horizontal")
    cbar_map.ax.xaxis.set_label_position("top")
    cbar_map.ax.xaxis.tick_top()
    cbar_map.set_label("Integrated Spectral Flux (Jy/beam·km/s)")

    cax_pv = fig.add_axes([0.08 + 0.445 + 0.03, 0.84, 0.445, 0.01])
    cbar_pv = fig.colorbar(im_pv, cax=cax_pv, orientation="horizontal")
    cbar_pv.ax.xaxis.set_label_position("top")
    cbar_pv.ax.xaxis.tick_top()
    cbar_pv.set_label("Intensity (Jy/beam)")

    if png_out:
        plt.savefig(png_out, dpi=600, bbox_inches="tight")
    plt.show()

    return pv


# ─────────────────────────────────────────────
#   Full pipeline
# ─────────────────────────────────────────────

def run_pv(
    fits_path: str,
    prefix: str,
    window: SpectralWindow,
    angle: float,
    length_pix: int,
    rms: float,
    ra: Optional[str] = None,
    dec: Optional[str] = None,
    source: Optional[SkyCoord] = None,
    kernel: int = 50,
    mask: bool = True,
    mask_mult: float = 3.0,
    sysvel: Optional[float] = None,
    pv_fits: Optional[str] = None,
    png_out: Optional[str] = None,
    contour_levels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Runs the full PV pipeline: moment-0 computation followed by PV extraction.

    This is the recommended entry point for interactive use in notebooks.
    It combines :func:`compute_moment0` and :func:`extract_pv` into a
    single call.

    Parameters
    ----------
    fits_path : str
        Path to the input FITS spectral cube.
    prefix : str
        Base name for the moment-0 output file.
    window : SpectralWindow
        Channel range to process.
    angle : float
        Position angle of the PV slice in degrees.
    length_pix : int
        Number of sample points along the spatial slice.
    rms : float
        RMS noise level in Jy/beam.
    ra : str, optional
        Right ascension of the slice center. Used when ``source`` is ``None``.
    dec : str, optional
        Declination of the slice center. Used when ``source`` is ``None``.
    source : SkyCoord, optional
        Source position. Overrides ``ra``/``dec``.
    kernel : int, optional
        RMS estimation kernel for bettermoments. Defaults to 50.
    mask : bool, optional
        Whether to apply a threshold mask to the moment-0. Defaults to ``True``.
    mask_mult : float, optional
        Mask threshold in units of RMS. Defaults to 3.0.
    sysvel : float, optional
        Systemic velocity (km/s). Reserved for future use.
    pv_fits : str, optional
        Output path for the PV FITS file.
    png_out : str, optional
        Output path for the figure PNG.
    contour_levels : ndarray, optional
        Contour level multipliers in units of ``rms``.

    Returns
    -------
    pv : ndarray, shape (length_pix, window.n_channels)
        Extracted PV array.

    Examples
    --------
    >>> from myastroutilities.structures import SpectralWindow
    >>> from myastroutilities.pv.extract import run_pv
    >>>
    >>> pv = run_pv(
    ...     fits_path="SiO_cube.fits",
    ...     prefix="UC1_SiO",
    ...     window=SpectralWindow(ch_min=930, ch_max=1020),
    ...     angle=126.0,
    ...     length_pix=200,
    ...     rms=9.809e-3,
    ...     ra="18h20m24.821s",
    ...     dec="-16d11m35.08s",
    ... )
    """
    mom0_path = compute_moment0(fits_path, prefix, window, kernel, mask, mask_mult)
    return extract_pv(
        cube_path=fits_path,
        mom0_path=mom0_path,
        window=window,
        angle=angle,
        length_pix=length_pix,
        rms=rms,
        source=source,
        ra=ra,
        dec=dec,
        sysvel=sysvel,
        pv_fits=pv_fits,
        png_out=png_out,
        contour_levels=contour_levels,
    )
