"""
Tests for myastroutilities.pv.geometry.

Verifies boundary detection, fraction computation, and coordinate
generation for PV slice geometry. These tests are particularly important
because these functions were previously copy-pasted in 6+ locations —
a regression here could silently affect all notebooks simultaneously.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

import numpy as np
import pytest

from myastroutilities.pv.geometry import (
    find_tmin_tmax,
    inside_fraction,
    max_symmetric_slice,
    slice_pixel_coords,
)

# Image size used throughout the tests
NX, NY = 512, 512


# ─────────────────────────────────────────────
#   find_tmin_tmax
# ─────────────────────────────────────────────

class TestFindTminTmax:
    def test_center_horizontal(self):
        # Horizontal slice through the center of the image
        t_min, t_max = find_tmin_tmax(256, 256, 0.0, NX, NY)
        assert t_min < 0 < t_max
        assert abs(t_min + t_max) < 1.0   # approximately symmetric

    def test_center_vertical(self):
        t_min, t_max = find_tmin_tmax(256, 256, np.pi / 2, NX, NY)
        assert t_min < 0 < t_max

    def test_center_diagonal(self):
        t_min, t_max = find_tmin_tmax(256, 256, np.deg2rad(45), NX, NY)
        assert t_min < 0 < t_max

    def test_range_is_ordered(self):
        t_min, t_max = find_tmin_tmax(200, 300, np.deg2rad(126), NX, NY)
        assert t_min <= t_max

    def test_center_outside_image(self):
        # Center far outside the image; both values should be 0
        t_min, t_max = find_tmin_tmax(-1000, -1000, 0.0, NX, NY)
        assert t_min == 0.0
        assert t_max == 0.0

    def test_pa_126_degrees(self):
        # PA=126° is the primary slice angle in the M17 SW / UC1 analysis
        angle = np.deg2rad(126)
        t_min, t_max = find_tmin_tmax(256, 256, angle, NX, NY)
        assert t_min < -100  # at least 100 pixels on each side
        assert t_max >  100


# ─────────────────────────────────────────────
#   inside_fraction
# ─────────────────────────────────────────────

class TestInsideFraction:
    def test_fully_inside(self):
        # Slice of 200 px; valid range spans ±310 px → fraction = 1
        assert inside_fraction(200, -310, 310) == pytest.approx(1.0)

    def test_partially_inside_right(self):
        # Slice of 200 px; valid range only covers [-100, 310]
        f = inside_fraction(200, -100, 310)
        assert 0.0 < f < 1.0

    def test_zero_overlap(self):
        # Valid range does not reach the slice
        assert inside_fraction(200, 150, 310) == pytest.approx(0.0)

    def test_zero_length(self):
        assert inside_fraction(0, -100, 100) == pytest.approx(0.0)

    def test_negative_length(self):
        assert inside_fraction(-10, -100, 100) == pytest.approx(0.0)


# ─────────────────────────────────────────────
#   max_symmetric_slice
# ─────────────────────────────────────────────

class TestMaxSymmetricSlice:
    def test_center_gives_positive_length(self):
        L = max_symmetric_slice(256, 256, np.deg2rad(126), NX, NY)
        assert L > 0

    def test_outside_image_gives_zero(self):
        L = max_symmetric_slice(-1000, -1000, 0.0, NX, NY)
        assert L == pytest.approx(0.0)

    def test_length_bounded_by_image(self):
        # Horizontal slice through center; max length ≤ image width
        L = max_symmetric_slice(256, 256, 0.0, NX, NY)
        assert L <= NX


# ─────────────────────────────────────────────
#   slice_pixel_coords
# ─────────────────────────────────────────────

class TestSlicePixelCoords:
    def test_output_shapes(self):
        x, y, s = slice_pixel_coords(256, 256, np.deg2rad(90), 200)
        assert x.shape == (200,)
        assert y.shape == (200,)
        assert s.shape == (200,)

    def test_center_is_at_midpoint(self):
        # The midpoint of the offset array should be ~0
        _, _, s = slice_pixel_coords(256, 256, 0.0, 200)
        assert abs(s[len(s) // 2]) < 1.0

    def test_horizontal_slice_constant_y(self):
        # angle = 0 → cos(0)=1, sin(0)=0 → y stays constant
        x, y, _ = slice_pixel_coords(256, 128, 0.0, 100)
        assert np.allclose(y, 128)

    def test_vertical_slice_constant_x(self):
        # angle = π/2 → cos=0, sin=1 → x stays constant
        x, y, _ = slice_pixel_coords(200, 256, np.pi / 2, 100)
        assert np.allclose(x, 200)

    def test_offsets_are_symmetric(self):
        _, _, s = slice_pixel_coords(256, 256, 0.0, 100)
        assert np.allclose(s, -s[::-1])
