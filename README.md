# myastroutilities

**A reusable Python toolkit for radio astronomy data analysis.**

Developed by **J. Grisales-Casadiegos** in the context of radio continuum and spectral line
analysis of H II regions and molecular clouds, this package consolidates
utilities that were previously scattered across multiple analysis repositories.
It is designed to be general enough for use in any radio interferometric data reduction
and analysis workflow.

---

## Motivation

Radio astronomy analysis notebooks and scripts tend to accumulate repeated patterns:
loading spectral cubes, extracting moment maps, computing position-velocity (PV) diagrams,
and plotting FITS images. These patterns are error-prone to duplicate and hard to maintain
when they live inside project-specific directories.

`myastroutilities` solves this by providing a single, installable, versioned package
that any project can depend on — avoiding copy-paste duplication and making it easy
for collaborators to replicate or extend the analysis.

---

## Installation

### From GitHub (recommended for users)

```bash
pip install git+https://github.com/jennifergc/myastroutilities.git
```

### Editable install for local development

```bash
git clone https://github.com/jennifergc/myastroutilities.git
cd myastroutilities
pip install -e ".[dev]"
```

> **Note for conda users:** If the editable install fails with
> `ModuleNotFoundError: No module named 'setuptools.backends'`, use:
>
> ```bash
> pip install --config-settings editable_mode=compat -e ".[dev]"
> ```
>
> This is a known limitation of the setuptools package distributed by conda,
> which omits the `backends` submodule. The workaround forces the legacy
> `.pth`-file editable install, which works identically.

---

## Package structure

```
myastroutilities/
├── structures.py          # Core data classes shared across all subpackages
├── cube/
│   ├── io.py              # FITS cube loading and metadata extraction
│   └── moments.py         # Moment-map computation via bettermoments
├── pv/
│   ├── geometry.py        # Position-velocity slice geometry
│   └── extract.py         # PV diagram extraction and visualization
└── plotting/
    └── fits_plotter.py    # FITS image visualization with overlays
```

---

## Quick start

```python
from myastroutilities.structures import UC1, SpectralWindow
from myastroutilities.cube.io import load_spectral_cube, get_image_metadata
from myastroutilities.cube.moments import run_bettermoments
from myastroutilities.pv.geometry import find_tmin_tmax, slice_pixel_coords
from myastroutilities.pv.extract import run_pv
from myastroutilities.plotting.fits_plotter import FITSPlotter

# Define a spectral window and load the cube
window = SpectralWindow(ch_min=930, ch_max=1020)
cube = load_spectral_cube("path/to/cube.fits")

# Get the pixel position of UC1
meta = get_image_metadata("path/to/cube.fits")
x0, y0 = UC1.pixel_coords(meta.wcs)

# Extract a PV diagram
pv = run_pv(
    fits_path="path/to/cube.fits",
    prefix="UC1",
    window=window,
    ra="18h20m24.821s",
    dec="-16d11m35.02s",
    angle=126.0,
    length_pix=200,
    rms=9.809e-3,
)

# Visualize a moment map
plotter = FITSPlotter("moment0.fits", moment="m0", ask=False)
plotter.plot(title="Moment 0 — M17 SW UC1")
```

---

## Module reference

### `myastroutilities.structures`

**Problem it solves:** Source coordinates, channel ranges, beam parameters, and moment
configurations were hard-coded repeatedly across notebooks and scripts, making them
fragile and inconsistent.

| Class / Object | Description |
|----------------|-------------|
| `Source` | Astronomical source with name, `SkyCoord`, and label. Provides `pixel_coords(wcs)`. |
| `SpectralWindow` | Channel range `[ch_min, ch_max]`. Provides `.n_channels`, `.slice`, and is iterable. |
| `BeamParams` | Synthesized beam parameters (BMAJ, BMIN, BPA in arcsec/degrees). Includes `from_header()` class method. |
| `ImageMetadata` | WCS, pixel scale, beam, and header bundled together. Produced by `get_image_metadata()`. |
| `MomentConfig` | Visual configuration (colormap, colorbar label, contour color) for a given moment type. |
| `UC1` | Pre-built `Source` constant for the UC1 hypercompact H II region (M17 SW). |
| `MOMENT_CONFIGS` | Dictionary mapping `'m0'`, `'m1'`, `'m2'`, `'continuo'` to their `MomentConfig`. |
| `get_moment_config(moment)` | Returns the appropriate `MomentConfig`, falling back to a default. |

---

### `myastroutilities.cube.io`

**Problem it solves:** Opening FITS cubes and converting their spectral axis to km/s
required 5–8 lines of boilerplate that appeared in 9+ locations across notebooks.
WCS and pixel-scale extraction was equally repetitive.

| Function | Description |
|----------|-------------|
| `open_fits(path)` | Opens a FITS file and returns `(data.squeeze(), header, WCS2D)`. |
| `load_spectral_cube(filename, ...)` | Loads a `SpectralCube` and converts the spectral axis to km/s via the radio velocity convention. |
| `get_subcube(filename, window, ...)` | Loads the cube and extracts the sub-cube defined by a `SpectralWindow`. Returns `(subcube, vel_axis)`. |
| `get_image_metadata(fits_path)` | Extracts WCS, pixel scale (arcsec/pixel), and beam parameters from a FITS file. Returns `ImageMetadata`. |
| `estimate_rms_from_channels(data, n_edge)` | Estimates the RMS noise using the outermost `n_edge` channels at each end of the cube. |

---

### `myastroutilities.cube.moments`

**Problem it solves:** Moment-map computation with `bettermoments` required knowledge
of internal API details and header manipulation that was copy-pasted across many cells.
The function signature was also inconsistent between scripts (missing `mask_mult`).

| Function | Description |
|----------|-------------|
| `run_bettermoments(ruta_fits, nombre_archivo, channel_min, channel_max, N, momento, mask, mask_mult)` | Computes moment 0 or 1 using bettermoments. Applies an optional RMS-threshold mask, writes the output FITS, and displays the map. Returns the output path. |

---

### `myastroutilities.pv.geometry`

**Problem it solves:** The functions `find_tmin_tmax` and `inside_fraction` were
copy-pasted **6 and 2 times**, respectively, inside a single notebook
(`N-06-PV_diagrams.ipynb`). Centralizing them eliminates the risk of diverging versions.

| Function | Description |
|----------|-------------|
| `find_tmin_tmax(xc, yc, angle_rad, nx, ny)` | Computes the parametric range `[t_min, t_max]` of a line through `(xc, yc)` that stays within the image boundaries. |
| `inside_fraction(L, t_min, t_max)` | Returns the fraction of a symmetric slice `[-L/2, L/2]` that falls within `[t_min, t_max]`. |
| `max_symmetric_slice(xc, yc, angle_rad, nx, ny)` | Returns the maximum length of a symmetric slice centered at `(xc, yc)` that fits entirely within the image. |
| `slice_pixel_coords(xc, yc, angle_rad, length_pix)` | Generates the pixel coordinates `(x_pix, y_pix, offsets)` along a slice of given length. |

---

### `myastroutilities.pv.extract`

**Problem it solves:** Three scripts (`pvutils.py`, `pv-gen.py`, `pv_slice_wcs.py`)
duplicated the PV extraction logic with minor variations. This module provides a single
canonical implementation covering both the extraction algorithm and the visualization.

| Function | Description |
|----------|-------------|
| `compute_moment0(fits_path, prefix, window, ...)` | Computes the moment-0 map for the given spectral window and returns the path to the output FITS. |
| `extract_pv(cube_path, mom0_path, ...)` | Extracts the PV array along a spatial slice and plots it alongside the moment-0 map. Returns the PV `ndarray`. |
| `run_pv(fits_path, prefix, window, ...)` | Full pipeline: computes moment 0 and extracts the PV diagram in a single call. |

---

### `myastroutilities.plotting.fits_plotter`

**Problem it solves:** Two diverging versions of `FITSPlotter` existed simultaneously
(`fits_plotter.py` and `fits_plotter_updated.py`). This module provides the unified,
canonical class with all features from both versions plus configurable PA angles.

| Class | Description |
|-------|-------------|
| `FITSPlotter` | Displays a base FITS image with optional contour overlay (reprojected), synthesized beam ellipses, UC1 marker, and position-angle lines. Supports both WCS-projection axes and ΔRA/ΔDec (arcsec) axes relative to UC1. Can be used programmatically (`ask=False`) or interactively (`ask=True`). |

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_fits` | required | Path to the base FITS image. |
| `contour_fits` | `None` | Path to a FITS file used for contour overlay (reprojected to base WCS). |
| `moment` | `None` | One of `'m0'`, `'m1'`, `'m2'`, `'continuo'`. Determines colormap and colorbar label. |
| `use_delta_coords` | `False` | If `True`, plots ΔRA/ΔDec axes in arcsec relative to UC1. |
| `show_uc1` | `True` | If `True`, draws the UC1 star marker and label. |
| `show_pa_lines` | `True` | If `True`, draws the position-angle lines. |
| `pa_angles` | `(126, 216)` | Tuple of position angles to draw (degrees). |
| `uc1_coord` | internal | Override the UC1 `SkyCoord` if working on a different source. |
| `ask` | `True` | If `True`, queries the user interactively at plot time. Set to `False` for scripts. |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Array operations |
| `astropy` | ≥ 5.3 | FITS I/O, WCS, coordinates, units |
| `matplotlib` | ≥ 3.7 | Visualization |
| `spectral-cube` | ≥ 0.6 | Spectral cube handling |
| `scipy` | ≥ 1.11 | `map_coordinates` for PV extraction |
| `reproject` | ≥ 0.13 | Contour reprojection in `FITSPlotter` |
| `bettermoments` | ≥ 1.0 | Moment-map computation |

---

## Authorship and citation

**Author:** J. Grisales-Casadiegos
**Repository:** https://github.com/jennifergc/myastroutilities
**License:** MIT

If this package contributes to published research, a citation or acknowledgment
in the methods section is appreciated:

> *The data reduction and analysis made use of* myastroutilities *(J. Grisales-Casadiegos, 2026),
> a Python toolkit for radio astronomy developed at Universidad de Guanajuato.*

---

## Contributing

Contributions, bug reports, and feature requests are welcome via
[GitHub Issues](https://github.com/jennifergc/myastroutilities/issues)
and pull requests.

Please ensure that new functions include NumPy-style docstrings written in
third-person voice and that corresponding tests are added under `tests/`.

---

## Roadmap

- [ ] Gaussian fitting along PV slices (porting `slices.py` / CASA workflow)
- [ ] Interactive PV slicer widget (Jupyter)
- [ ] Multi-source support (extend beyond UC1 defaults)
- [ ] Sphinx documentation hosted on Read the Docs
- [ ] PyPI release
