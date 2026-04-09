"""
myastroutilities.cube.moments
==============================
Moment-map computation via the ``bettermoments`` library.

Problem solved
--------------
The ``run_bettermoments`` function was defined once in ``run_moments.py``
but called with inconsistent signatures (missing ``mask_mult``) across
multiple scripts and notebooks. This module provides a single, clean
implementation with full parameter documentation and a return value
(output path) that was previously missing.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import bettermoments as bm


def run_bettermoments(
    ruta_fits: str,
    nombre_archivo: str,
    channel_min: int,
    channel_max: int,
    N: int,
    momento: str,
    mask: bool,
    mask_mult: float = 3.0,
) -> str:
    """
    Computes and saves a moment-0 or moment-1 map using ``bettermoments``.

    Loads the spectral cube, estimates the RMS noise, optionally applies
    a threshold mask, computes the requested moment, saves the result as
    a FITS file, and displays the map inline.

    Parameters
    ----------
    ruta_fits : str
        Path to the input FITS spectral cube.
    nombre_archivo : str
        Base name for the output file (without extension).
        The output is saved next to the input cube as
        ``<nombre_archivo>_momento{0|1}_BM.fits``.
    channel_min : int
        First channel of the integration range (inclusive).
    channel_max : int
        Last channel of the integration range (inclusive).
    N : int
        Number of channels used by ``bettermoments`` to estimate the RMS.
        A larger value gives a more robust estimate.
    momento : str
        ``'0'`` for moment 0 (integrated intensity),
        ``'1'`` for moment 1 (intensity-weighted mean velocity).
    mask : bool
        If ``True``, applies a threshold mask at ``mask_mult × RMS``
        before computing the moment.
    mask_mult : float, optional
        Threshold multiplier for the mask in units of RMS. Defaults to 3.0.

    Returns
    -------
    nombre_salida : str
        Absolute path to the saved FITS file.

    Raises
    ------
    ValueError
        Raised when ``momento`` is not ``'0'`` or ``'1'``.

    Examples
    --------
    >>> path = run_bettermoments(
    ...     "SiO_cube.fits", "SiO_UC1",
    ...     channel_min=930, channel_max=1020,
    ...     N=50, momento="0", mask=True, mask_mult=3.0,
    ... )
    >>> print(path)
    '/data/SiO_UC1_momento0_BM.fits'
    """
    ruta_salida = os.path.dirname(ruta_fits)

    # --- Load cube and header ---
    data, _ = bm.load_cube(ruta_fits)
    with fits.open(ruta_fits) as hdul:
        header = hdul[0].header.copy()

    # --- Build velocity axis in km/s ---
    crval3 = header.get("CRVAL3")
    cdelt3 = header.get("CDELT3")
    crpix3 = header.get("CRPIX3", 1.0)
    rest_freq = header.get("RESTFRQ")
    n_channels = data.shape[0]

    velax_hz = crval3 + (np.arange(n_channels) - (crpix3 - 1)) * cdelt3
    c = 299792.458  # km/s
    velax_kms = c * (rest_freq - velax_hz) / rest_freq

    # --- Select channels ---
    data_selected = data[channel_min : channel_max + 1, :, :]
    velax_selected = velax_kms[channel_min : channel_max + 1]

    # --- Estimate RMS ---
    sigma_rms = bm.estimate_RMS(data, N=N)
    print(f"Estimated RMS: {sigma_rms:.3e} (data units)")

    # --- Apply mask ---
    if mask:
        data_mask = bm.get_threshold_mask(
            data_selected, clip=mask_mult, rms=sigma_rms, noise_channels=5
        )
        data_masked = np.where(data_mask, data_selected, 0)
    else:
        data_masked = data_selected

    # --- Compute moment ---
    if momento == "0":
        moment, err = bm.collapse_zeroth(velax_selected, data_masked, sigma_rms)
        header["BUNIT"] = "Jy/beam km/s"
        titulo = "Moment 0 Map"
        etiqueta = "Integrated Intensity (Jy/beam km/s)"
        colormap = "inferno"
        nombre_salida = os.path.join(ruta_salida, f"{nombre_archivo}_momento0_BM.fits")

    elif momento == "1":
        moment, err = bm.collapse_first(velax_selected, data_masked, sigma_rms)
        header["CTYPE3"] = "VELO"
        header["CUNIT3"] = "km/s"
        header["BUNIT"] = "km/s"
        titulo = "Moment 1 Map (Mean Velocity)"
        etiqueta = "Velocity (km/s)"
        colormap = "jet"
        nombre_salida = os.path.join(ruta_salida, f"{nombre_archivo}_momento1_BM.fits")

    else:
        raise ValueError(
            f"Argument 'momento' must be '0' or '1', received '{momento}'."
        )

    err_median = np.nanmedian(err)
    print(f"Median error of Moment {momento}: {err_median:.3e}")

    # --- Remove spectral axis keywords ---
    for key in ["NAXIS3", "CRPIX3", "CDELT3", "CRVAL3", "CTYPE3"]:
        header.remove(key, ignore_missing=True, remove_all=True)

    # --- Save FITS ---
    fits.PrimaryHDU(moment, header=header).writeto(nombre_salida, overwrite=True)
    print(f"Output written to: {nombre_salida}")

    # --- Display map ---
    plt.figure(figsize=(8, 6))
    plt.imshow(moment, origin="lower", cmap=colormap)
    plt.colorbar(label=etiqueta)
    plt.title(titulo)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.show()

    return nombre_salida
