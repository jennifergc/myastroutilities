"""
Tests for myastroutilities.structures.

Verifies that the core data classes behave correctly for the cases most
likely to cause regressions: construction, property derivation, header
parsing, and the UC1 constant.

Author
------
J. Grisales-Casadiegos
https://github.com/jennifergc/myastroutilities
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header

from myastroutilities.structures import (
    Source,
    SpectralWindow,
    BeamParams,
    MomentConfig,
    MOMENT_CONFIGS,
    get_moment_config,
    UC1,
)


# ─────────────────────────────────────────────
#   Source
# ─────────────────────────────────────────────

class TestSource:
    def test_construction(self):
        coord = SkyCoord(ra="18h20m24.821s", dec="-16d11m35.02s", frame="icrs")
        src = Source(name="UC1", coord=coord, label="UC1")
        assert src.name == "UC1"
        assert src.label == "UC1"

    def test_default_label(self):
        coord = SkyCoord(ra="0h0m0s", dec="0d0m0s", frame="icrs")
        src = Source(name="test", coord=coord)
        assert src.label == ""

    def test_uc1_constant_coordinates(self):
        assert UC1.name == "UC1"
        ra_deg = UC1.coord.ra.deg
        dec_deg = UC1.coord.dec.deg
        # RA ≈ 275.103° (18h20m24.821s)
        assert abs(ra_deg - 275.103) < 0.01
        # Dec ≈ -16.193° (-16d11m35.02s)
        assert abs(dec_deg - (-16.193)) < 0.01


# ─────────────────────────────────────────────
#   SpectralWindow
# ─────────────────────────────────────────────

class TestSpectralWindow:
    def test_n_channels(self):
        w = SpectralWindow(ch_min=930, ch_max=1020)
        assert w.n_channels == 91

    def test_single_channel(self):
        w = SpectralWindow(ch_min=500, ch_max=500)
        assert w.n_channels == 1

    def test_slice(self):
        w = SpectralWindow(ch_min=10, ch_max=20)
        s = w.slice
        arr = np.arange(50)
        assert list(arr[s]) == list(range(10, 21))

    def test_iteration(self):
        w = SpectralWindow(ch_min=5, ch_max=8)
        assert list(w) == [5, 6, 7, 8]

    def test_len(self):
        w = SpectralWindow(ch_min=0, ch_max=99)
        assert len(w) == 100


# ─────────────────────────────────────────────
#   BeamParams
# ─────────────────────────────────────────────

class TestBeamParams:
    def test_from_header(self):
        hdr = Header()
        hdr["BMAJ"] = 1.0 / 3600.0   # 1 arcsec in degrees
        hdr["BMIN"] = 0.5 / 3600.0   # 0.5 arcsec in degrees
        hdr["BPA"]  = 45.0
        beam = BeamParams.from_header(hdr)
        assert beam is not None
        assert abs(beam.bmaj - 1.0) < 1e-6
        assert abs(beam.bmin - 0.5) < 1e-6
        assert beam.bpa == 45.0

    def test_from_header_missing_keyword(self):
        hdr = Header()
        hdr["BMAJ"] = 1.0 / 3600.0
        # BMIN and BPA missing
        beam = BeamParams.from_header(hdr)
        assert beam is None

    def test_as_dict(self):
        beam = BeamParams(bmaj=2.0, bmin=1.0, bpa=30.0)
        d = beam.as_dict()
        assert d == {"bmaj": 2.0, "bmin": 1.0, "bpa": 30.0}


# ─────────────────────────────────────────────
#   MomentConfig and MOMENT_CONFIGS
# ─────────────────────────────────────────────

class TestMomentConfig:
    def test_all_keys_present(self):
        for key in ("m0", "m1", "m2", "continuo"):
            assert key in MOMENT_CONFIGS

    def test_get_moment_config_known(self):
        cfg = get_moment_config("m0")
        assert isinstance(cfg, MomentConfig)
        assert "Jy/beam" in cfg.label or "km/s" in cfg.label

    def test_get_moment_config_none(self):
        cfg = get_moment_config(None)
        assert isinstance(cfg, MomentConfig)

    def test_get_moment_config_unknown(self):
        cfg = get_moment_config("unknown_moment")
        assert isinstance(cfg, MomentConfig)

    def test_m1_uses_jet_colormap(self):
        assert MOMENT_CONFIGS["m1"].colormap == "jet"

    def test_m0_contour_color(self):
        assert MOMENT_CONFIGS["m0"].contour_color == "white"
