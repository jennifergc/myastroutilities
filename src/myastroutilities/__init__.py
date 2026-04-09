"""
myastroutilities
================
A reusable Python toolkit for radio astronomy data analysis.

Developed by Jennifer Grisales, this package consolidates utilities for
spectral cube handling, moment-map computation, position-velocity diagram
extraction, and FITS image visualization.

Subpackages
-----------
structures
    Core data classes shared across all subpackages.
cube
    FITS cube I/O, metadata extraction, and moment-map computation.
pv
    Position-velocity slice geometry and diagram extraction.
plotting
    FITS image visualization with contour overlays and beam rendering.

Typical usage
-------------
>>> from myastroutilities.structures import UC1, SpectralWindow
>>> from myastroutilities.cube.io import load_spectral_cube
>>> from myastroutilities.pv.extract import run_pv
>>> from myastroutilities.plotting.fits_plotter import FITSPlotter
"""

__version__ = "0.1.0"
__author__ = "J. Grisales-Casadiegos"
__license__ = "MIT"
__url__ = "https://github.com/jennifergc/myastroutilities"
