"""
myastroutilities.cube.io
========================
FITS cube loading, spectral-axis conversion, spatial metadata extraction,
and RMS noise estimation.

Problem solved
--------------
The following boilerplate appeared in 9+ locations across notebooks and scripts:

    cube = SpectralCube.read(filename)
    rest_freq = cube.header['RESTFRQ'] * u.Hz
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                                   rest_value=rest_freq)

Similarly, WCS and pixel-scale extraction required 4–6 repetitive lines.
This module centralizes those patterns behind simple, documented functions.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from spectral_cube import SpectralCube

from myastroutilities.structures import BeamParams, ImageMetadata, SpectralWindow


# ─────────────────────────────────────────────
#   Basic FITS I/O
# ─────────────────────────────────────────────

def open_fits(path: str) -> Tuple[np.ndarray, fits.Header, WCS]:
    """
    Opens a FITS file and returns the data, header, and 2-D celestial WCS.

    Reduces the data array with ``squeeze()`` to remove degenerate axes
    (e.g., the Stokes axis present in CASA-produced cubes).

    Parameters
    ----------
    path : str
        Absolute or relative path to the FITS file.

    Returns
    -------
    data : ndarray
        Image data with degenerate axes removed.
    header : astropy.io.fits.Header
        Primary FITS header.
    wcs : WCS
        Two-dimensional celestial WCS built from the header.

    Examples
    --------
    >>> data, hdr, wcs = open_fits("moment0.fits")
    >>> data.shape
    (512, 512)
    """
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data.squeeze()
    wcs = WCS(header, naxis=2)
    return data, header, wcs


# ─────────────────────────────────────────────
#   Spectral cube loading
# ─────────────────────────────────────────────

def load_spectral_cube(
    filename: str,
    rest_freq_key: str = "RESTFRQ",
    velocity_convention: str = "radio",
) -> SpectralCube:
    """
    Loads a FITS spectral cube and converts its spectral axis to km/s.

    Problem solved: loading a cube without this function requires knowing
    the bettermoments / spectral-cube API and repeating the rest-frequency
    retrieval and unit conversion every time.

    Parameters
    ----------
    filename : str
        Path to the FITS spectral cube.
    rest_freq_key : str, optional
        FITS header keyword containing the rest frequency in Hz.
        Defaults to ``'RESTFRQ'``.
    velocity_convention : str, optional
        Velocity convention to apply. One of ``'radio'`` or ``'optical'``.
        Defaults to ``'radio'``.

    Returns
    -------
    SpectralCube
        Cube with the spectral axis expressed in km/s.

    Examples
    --------
    >>> cube = load_spectral_cube("SiO_cube.fits")
    >>> cube.spectral_axis[:3]
    <Quantity [19.3, 19.5, 19.7] km / s>
    """
    cube = SpectralCube.read(filename)
    rest_freq = cube.header[rest_freq_key] * u.Hz
    return cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention=velocity_convention,
        rest_value=rest_freq,
    )


def get_subcube(
    filename: str,
    window: SpectralWindow,
    rest_freq_key: str = "RESTFRQ",
) -> Tuple[SpectralCube, np.ndarray]:
    """
    Loads a spectral cube and extracts the sub-cube defined by *window*.

    Combines :func:`load_spectral_cube` with channel slicing to reduce
    memory footprint when only a subset of channels is needed.

    Parameters
    ----------
    filename : str
        Path to the FITS spectral cube.
    window : SpectralWindow
        Channel range to extract.
    rest_freq_key : str, optional
        Header keyword for the rest frequency. Defaults to ``'RESTFRQ'``.

    Returns
    -------
    subcube : SpectralCube
        Sub-cube in km/s for the specified channel range.
    vel_axis : ndarray
        Velocity axis values in km/s, shape ``(window.n_channels,)``.

    Examples
    --------
    >>> window = SpectralWindow(ch_min=930, ch_max=1020)
    >>> subcube, vel = get_subcube("cube.fits", window)
    >>> vel.shape
    (91,)
    """
    cube = load_spectral_cube(filename, rest_freq_key)
    subcube = cube[window.slice]
    vel_axis = subcube.spectral_axis.to_value("km/s")
    return subcube, vel_axis


# ─────────────────────────────────────────────
#   Image metadata
# ─────────────────────────────────────────────

def get_image_metadata(fits_path: str) -> ImageMetadata:
    """
    Extracts the spatial metadata from a FITS image.

    Reads the WCS, pixel scale (arcsec/pixel), and synthesized beam
    parameters from the FITS header and bundles them into an
    :class:`~myastroutilities.structures.ImageMetadata` object.

    Problem solved: these three pieces of information were extracted
    independently in 11+ locations, each with slightly different code.

    Parameters
    ----------
    fits_path : str
        Path to the FITS image.

    Returns
    -------
    ImageMetadata
        Object containing ``wcs``, ``pixel_scale_arcsec``, ``beam``,
        and ``header``.

    Examples
    --------
    >>> meta = get_image_metadata("moment0.fits")
    >>> meta.pixel_scale_arcsec
    0.04166...
    >>> x0, y0 = meta.wcs.world_to_pixel(coord)
    """
    with fits.open(fits_path) as hdul:
        header = hdul[0].header.copy()
    wcs = WCS(header, naxis=2)
    pixel_scale = abs(header.get("CDELT1", 1.0 / 3600.0)) * 3600.0
    beam = BeamParams.from_header(header)
    return ImageMetadata(wcs=wcs, pixel_scale_arcsec=pixel_scale,
                         beam=beam, header=header)


# ─────────────────────────────────────────────
#   RMS estimation
# ─────────────────────────────────────────────

def estimate_rms_from_channels(
    data: np.ndarray,
    n_edge: int = 5,
) -> float:
    """
    Estimates the RMS noise using the outermost channels of a spectral cube.

    Assumes that the first and last ``n_edge`` channels are free of
    significant line emission, making them suitable for noise estimation.

    Parameters
    ----------
    data : ndarray, shape (nchan, ny, nx)
        Spectral cube data array.
    n_edge : int, optional
        Number of channels to use at each end of the cube. Defaults to 5.

    Returns
    -------
    rms : float
        Standard deviation computed over the edge channels, ignoring NaNs.

    Examples
    --------
    >>> rms = estimate_rms_from_channels(cube_data, n_edge=10)
    >>> print(f"RMS = {rms:.3e} Jy/beam")
    """
    edge = np.concatenate([data[:n_edge], data[-n_edge:]], axis=0)
    return float(np.nanstd(edge))
