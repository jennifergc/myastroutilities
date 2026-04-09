# Changelog

All notable changes to `myastroutilities` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-04-08

### Added
- `myastroutilities.structures`: core data classes (`Source`, `SpectralWindow`,
  `BeamParams`, `ImageMetadata`, `MomentConfig`) and the `UC1` source constant.
- `myastroutilities.cube.io`: FITS cube loading (`open_fits`, `load_spectral_cube`,
  `get_subcube`, `get_image_metadata`) and RMS estimation (`estimate_rms_from_channels`).
- `myastroutilities.cube.moments`: moment-map computation via bettermoments
  (`run_bettermoments`).
- `myastroutilities.pv.geometry`: position-velocity slice geometry utilities
  (`find_tmin_tmax`, `inside_fraction`, `max_symmetric_slice`, `slice_pixel_coords`).
- `myastroutilities.pv.extract`: PV diagram extraction and visualization
  (`extract_pv`, `compute_moment0`, `run_pv`).
- `myastroutilities.plotting.fits_plotter`: `FITSPlotter` class for FITS image
  visualization with contour overlays, beam rendering, and UC1 marker.
- MIT License.
- Full NumPy-style docstrings throughout.
- Basic unit tests for structures, PV geometry, and cube I/O utilities.
