"""
myastroutilities.structures
===========================
Core data classes shared across all subpackages of myastroutilities.

Problem solved
--------------
Source coordinates, channel ranges, beam parameters, and moment-map
configurations were hard-coded repeatedly across notebooks and scripts,
creating fragile and inconsistent duplicates. This module centralizes
those definitions as typed, self-documenting data classes.

Author
------
Jennifer Grisales
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io.fits import Header


# ─────────────────────────────────────────────
#   Source
# ─────────────────────────────────────────────

@dataclass
class Source:
    """
    Represents an astronomical source with a name and sky coordinates.

    Attributes
    ----------
    name : str
        Identifier of the source (e.g., ``'UC1'``).
    coord : SkyCoord
        Sky position of the source in ICRS frame.
    label : str, optional
        Human-readable label for plot annotations.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> src = Source(name="UC1", coord=SkyCoord(ra="18h20m24.821s",
    ...              dec="-16d11m35.02s", frame="icrs"), label="UC1")
    >>> x, y = src.pixel_coords(wcs_2d)
    """

    name: str
    coord: SkyCoord
    label: str = ""

    def pixel_coords(self, wcs: WCS) -> tuple[float, float]:
        """
        Returns the pixel position of the source in the given WCS.

        Parameters
        ----------
        wcs : WCS
            Two-dimensional celestial WCS of the target image.

        Returns
        -------
        x_pix, y_pix : float
            Pixel coordinates (column, row) of the source.
        """
        return wcs.world_to_pixel(self.coord)


# Pre-built constant: UC1 hypercompact H II region in M17 SW
UC1 = Source(
    name="UC1",
    coord=SkyCoord(ra="18h20m24.821s", dec="-16d11m35.02s", frame="icrs"),
    label="UC1",
)
"""
Pre-built :class:`Source` constant for the UC1 hypercompact H II region in M17 SW.

Coordinates (ICRS): RA = 18h 20m 24.821s, Dec = -16° 11′ 35.02″.
Used as the reference position throughout the M17 SW / UC1 analysis.
"""


# ─────────────────────────────────────────────
#   SpectralWindow
# ─────────────────────────────────────────────

@dataclass
class SpectralWindow:
    """
    Defines a contiguous channel range within a spectral cube.

    Problem solved: ``ch_min`` and ``ch_max`` appeared as loose integer
    variables in 78 locations across notebooks. This class bundles them
    with derived properties and makes the intent explicit.

    Attributes
    ----------
    ch_min : int
        First channel of the window (inclusive).
    ch_max : int
        Last channel of the window (inclusive).

    Examples
    --------
    >>> w = SpectralWindow(ch_min=930, ch_max=1020)
    >>> w.n_channels
    91
    >>> data[w.slice]  # equivalent to data[930:1021]
    """

    ch_min: int
    ch_max: int

    @property
    def n_channels(self) -> int:
        """Returns the total number of channels in the window."""
        return self.ch_max - self.ch_min + 1

    @property
    def slice(self) -> slice:
        """Returns a Python ``slice`` object for array indexing."""
        return slice(self.ch_min, self.ch_max + 1)

    def __iter__(self):
        """Iterates over the channel indices in the window."""
        return iter(range(self.ch_min, self.ch_max + 1))

    def __len__(self) -> int:
        return self.n_channels


# ─────────────────────────────────────────────
#   BeamParams
# ─────────────────────────────────────────────

@dataclass
class BeamParams:
    """
    Stores the synthesized beam parameters extracted from a FITS header.

    Attributes
    ----------
    bmaj : float
        Major axis of the beam in arcseconds.
    bmin : float
        Minor axis of the beam in arcseconds.
    bpa : float
        Position angle of the beam in degrees.

    Examples
    --------
    >>> with fits.open("image.fits") as hdul:
    ...     beam = BeamParams.from_header(hdul[0].header)
    """

    bmaj: float
    bmin: float
    bpa: float

    @classmethod
    def from_header(cls, header: Header) -> Optional["BeamParams"]:
        """
        Constructs a :class:`BeamParams` instance from a FITS header.

        Reads ``BMAJ``, ``BMIN`` (degrees → arcseconds), and ``BPA``
        (degrees) from the header. Returns ``None`` if any keyword is absent.

        Parameters
        ----------
        header : astropy.io.fits.Header
            FITS header containing beam keywords.

        Returns
        -------
        BeamParams or None
        """
        try:
            return cls(
                bmaj=header["BMAJ"] * 3600.0,
                bmin=header["BMIN"] * 3600.0,
                bpa=header["BPA"],
            )
        except KeyError:
            return None

    def as_dict(self) -> dict:
        """Returns the beam parameters as a plain dictionary."""
        return {"bmaj": self.bmaj, "bmin": self.bmin, "bpa": self.bpa}


# ─────────────────────────────────────────────
#   ImageMetadata
# ─────────────────────────────────────────────

@dataclass
class ImageMetadata:
    """
    Bundles the spatial metadata of a FITS image.

    Problem solved: WCS extraction, pixel-scale computation, and beam
    retrieval required 5–8 lines of boilerplate that appeared in 11+
    locations. This class groups them into a single reusable object
    returned by :func:`~myastroutilities.cube.io.get_image_metadata`.

    Attributes
    ----------
    wcs : WCS
        Two-dimensional celestial WCS of the image.
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel (absolute value of CDELT1 × 3600).
    beam : BeamParams or None
        Synthesized beam parameters, or ``None`` if not present in the header.
    header : astropy.io.fits.Header
        Original FITS header.
    """

    wcs: WCS
    pixel_scale_arcsec: float
    beam: Optional[BeamParams]
    header: Header

    @property
    def shape(self) -> tuple[int, int]:
        """Returns ``(ny, nx)`` from the header NAXIS keywords."""
        return (self.header.get("NAXIS2"), self.header.get("NAXIS1"))


# ─────────────────────────────────────────────
#   MomentConfig
# ─────────────────────────────────────────────

@dataclass
class MomentConfig:
    """
    Visual and unit configuration for a specific moment-map type.

    Problem solved: Colormap names, colorbar labels, and overlay colors
    were duplicated in 40+ locations across notebooks and both versions
    of FITSPlotter. This class centralizes them.

    Attributes
    ----------
    label : str
        Colorbar label describing the physical quantity and unit.
    unit : str
        Physical unit string (e.g., ``'Jy/beam km/s'``).
    colormap : str
        Matplotlib colormap name.
    contour_color : str
        Color for contour lines drawn on top of this moment map.
    star_color : str
        Color for the UC1 marker drawn on top of this moment map.
    """

    label: str
    unit: str
    colormap: str
    contour_color: str = "white"
    star_color: str = "yellow"


MOMENT_CONFIGS: dict[str, MomentConfig] = {
    "m0": MomentConfig(
        label="Integrated Flux (Jy/beam km/s)",
        unit="Jy/beam km/s",
        colormap="gnuplot2",
        contour_color="white",
        star_color="lawngreen",
    ),
    "m1": MomentConfig(
        label="Velocity (km/s)",
        unit="km/s",
        colormap="jet",
        contour_color="black",
        star_color="fuchsia",
    ),
    "m2": MomentConfig(
        label="Velocity Dispersion (km/s)",
        unit="km/s",
        colormap="jet",
        contour_color="black",
        star_color="fuchsia",
    ),
    "continuo": MomentConfig(
        label="Intensity (Jy/beam)",
        unit="Jy/beam",
        colormap="gnuplot2",
        contour_color="white",
        star_color="lawngreen",
    ),
}
"""Pre-built :class:`MomentConfig` objects for standard moment types."""

_DEFAULT_MOMENT_CONFIG = MomentConfig(
    label="Intensity (Jy/beam)",
    unit="Jy/beam",
    colormap="gnuplot2",
    contour_color="white",
    star_color="yellow",
)


def get_moment_config(moment: Optional[str]) -> MomentConfig:
    """
    Returns the :class:`MomentConfig` for the given moment type.

    Falls back to a default configuration when ``moment`` is ``None``
    or not found in :data:`MOMENT_CONFIGS`.

    Parameters
    ----------
    moment : str or None
        One of ``'m0'``, ``'m1'``, ``'m2'``, ``'continuo'``, or ``None``.

    Returns
    -------
    MomentConfig
    """
    if moment is None:
        return _DEFAULT_MOMENT_CONFIG
    return MOMENT_CONFIGS.get(moment, _DEFAULT_MOMENT_CONFIG)
