"""
Tests for myastroutilities.cube.io.

Verifies the utility functions for FITS loading and metadata extraction
using synthetic in-memory FITS data, without requiring real data files.

Author
------
Jennifer Grisales
https://github.com/jennifergc/myastroutilities
"""

import io
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from myastroutilities.structures import SpectralWindow
from myastroutilities.cube.io import estimate_rms_from_channels


# ─────────────────────────────────────────────
#   estimate_rms_from_channels
# ─────────────────────────────────────────────

class TestEstimateRms:
    def test_gaussian_noise(self):
        rng = np.random.default_rng(42)
        # Cube: 100 channels, 32×32 pixels, Gaussian noise σ=0.01
        sigma_true = 0.01
        data = rng.normal(0, sigma_true, size=(100, 32, 32))
        rms = estimate_rms_from_channels(data, n_edge=10)
        # Estimated RMS should be within 20% of the true value
        assert abs(rms - sigma_true) / sigma_true < 0.20

    def test_uses_only_edge_channels(self):
        # Fill inner channels with a large signal; edges should be noise-free
        rng = np.random.default_rng(0)
        sigma_true = 0.005
        data = np.ones((100, 16, 16)) * 100.0   # strong signal everywhere
        data[:5]  = rng.normal(0, sigma_true, size=(5, 16, 16))
        data[-5:] = rng.normal(0, sigma_true, size=(5, 16, 16))
        rms = estimate_rms_from_channels(data, n_edge=5)
        # Should reflect the edge noise, not the inner signal
        assert rms < 1.0

    def test_with_nans(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0, 0.02, size=(50, 20, 20))
        data[0, :5, :5] = np.nan   # introduce NaNs in edge channels
        rms = estimate_rms_from_channels(data, n_edge=5)
        assert np.isfinite(rms)
        assert rms > 0


# ─────────────────────────────────────────────
#   SpectralWindow integration
# ─────────────────────────────────────────────

class TestSpectralWindowSlicing:
    def test_slice_on_array(self):
        data = np.arange(200)
        w = SpectralWindow(ch_min=10, ch_max=29)
        assert len(data[w.slice]) == w.n_channels == 20

    def test_iteration_matches_slice(self):
        w = SpectralWindow(ch_min=5, ch_max=9)
        from_iter = list(w)
        data = np.zeros(20)
        from_slice = list(range(*w.slice.indices(20)))
        assert from_iter == from_slice
