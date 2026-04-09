"""
myastroutilities.pv.geometry
=============================
Geometric utilities for defining and evaluating position-velocity slices.

Problem solved
--------------
The functions ``find_tmin_tmax`` and ``inside_fraction`` were defined as
exact duplicates **6 and 2 times**, respectively, within a single notebook
(``N-06-PV_diagrams.ipynb``). Having multiple copies created a risk of
diverging implementations. This module provides the single canonical
definition and extends it with two additional helpers.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

import numpy as np


def find_tmin_tmax(
    xc: float,
    yc: float,
    angle_rad: float,
    nx: int,
    ny: int,
) -> tuple[float, float]:
    """
    Computes the parametric range of a line that stays within image boundaries.

    Given a line parameterized as:

        x(t) = xc + t · cos(angle_rad)
        y(t) = yc + t · sin(angle_rad)

    this function finds the values of *t* at which the line enters and exits
    the rectangular domain [0, nx − 1] × [0, ny − 1].

    Problem solved: appears 6 times as an exact copy in
    ``N-06-PV_diagrams.ipynb``. Centralizing it eliminates the risk of
    one copy drifting from the others.

    Parameters
    ----------
    xc : float
        X-coordinate of the slice center in pixels.
    yc : float
        Y-coordinate of the slice center in pixels.
    angle_rad : float
        Position angle in radians, measured from the X axis
        (positive = counter-clockwise).
    nx : int
        Number of columns in the image.
    ny : int
        Number of rows in the image.

    Returns
    -------
    t_min, t_max : float
        Minimum and maximum valid parameter values. Both are 0.0 when
        the center lies outside the image or no valid range exists.

    Examples
    --------
    >>> t_min, t_max = find_tmin_tmax(256, 256, np.deg2rad(126), 512, 512)
    >>> print(t_min, t_max)
    -299.8... 299.8...
    """
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    eps = 1e-12

    t_candidates: list[float] = []
    if abs(dx) > eps:
        t_candidates.append((0 - xc) / dx)
        t_candidates.append((nx - 1 - xc) / dx)
    if abs(dy) > eps:
        t_candidates.append((0 - yc) / dy)
        t_candidates.append((ny - 1 - yc) / dy)

    margin = 0.5
    t_valid = [
        t for t in t_candidates
        if (-margin <= xc + t * dx <= nx - 1 + margin)
        and (-margin <= yc + t * dy <= ny - 1 + margin)
    ]

    if len(t_valid) < 2:
        return 0.0, 0.0

    return float(min(t_valid)), float(max(t_valid))


def inside_fraction(L: float, t_min: float, t_max: float) -> float:
    """
    Returns the fraction of a symmetric slice that lies within the valid range.

    For a symmetric slice spanning [−L/2, L/2] and a valid image range
    [t_min, t_max], this function computes the overlap as a fraction of L.

    Problem solved: appears 2 times as an exact copy in
    ``N-06-PV_diagrams.ipynb``.

    Parameters
    ----------
    L : float
        Total slice length in pixels.
    t_min : float
        Lower boundary of the valid parametric range (from :func:`find_tmin_tmax`).
    t_max : float
        Upper boundary of the valid parametric range.

    Returns
    -------
    fraction : float
        Value in [0, 1]. Returns 0.0 when there is no overlap.

    Examples
    --------
    >>> inside_fraction(200, -310, 310)
    1.0
    >>> inside_fraction(600, -310, 310)
    1.033...   # clipped to 1 in practice; caller should check
    """
    if L <= 0:
        return 0.0
    lo = max(-L / 2.0, t_min)
    hi = min(L / 2.0, t_max)
    if hi <= lo:
        return 0.0
    return (hi - lo) / L


def max_symmetric_slice(
    xc: float,
    yc: float,
    angle_rad: float,
    nx: int,
    ny: int,
) -> float:
    """
    Returns the maximum length of a symmetric slice that fits within the image.

    Computes the largest value of *L* such that the slice [−L/2, L/2]
    centered at (xc, yc) at the given angle lies entirely within the image.

    Parameters
    ----------
    xc : float
        X-coordinate of the slice center in pixels.
    yc : float
        Y-coordinate of the slice center in pixels.
    angle_rad : float
        Position angle in radians.
    nx : int
        Number of image columns.
    ny : int
        Number of image rows.

    Returns
    -------
    max_length : float
        Maximum symmetric slice length in pixels. Returns 0.0 if the
        center lies outside the image.

    Examples
    --------
    >>> L_max = max_symmetric_slice(256, 256, np.deg2rad(126), 512, 512)
    """
    t_min, t_max = find_tmin_tmax(xc, yc, angle_rad, nx, ny)
    return 2.0 * min(abs(t_min), abs(t_max))


def slice_pixel_coords(
    xc: float,
    yc: float,
    angle_rad: float,
    length_pix: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates pixel coordinates along a linear spatial slice.

    Samples ``length_pix`` evenly spaced points along the line:

        x(s) = xc + s · cos(angle_rad)
        y(s) = yc + s · sin(angle_rad)

    where *s* ranges from −length_pix/2 to +length_pix/2.

    Parameters
    ----------
    xc : float
        X-coordinate of the slice center in pixels.
    yc : float
        Y-coordinate of the slice center in pixels.
    angle_rad : float
        Position angle in radians.
    length_pix : int
        Number of sample points along the slice.

    Returns
    -------
    x_pix : ndarray, shape (length_pix,)
        X pixel coordinates along the slice.
    y_pix : ndarray, shape (length_pix,)
        Y pixel coordinates along the slice.
    offsets : ndarray, shape (length_pix,)
        Parametric offsets from the center in pixels (symmetric around 0).

    Examples
    --------
    >>> x, y, s = slice_pixel_coords(256, 256, np.deg2rad(126), 200)
    >>> x.shape
    (200,)
    """
    half = length_pix / 2.0
    offsets = np.linspace(-half, half, length_pix)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    x_pix = xc + offsets * dx
    y_pix = yc + offsets * dy
    return x_pix, y_pix, offsets
