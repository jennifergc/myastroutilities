"""
myastroutilities.plotting.fits_plotter
=======================================
FITS image visualization with contour overlays, synthesized beam rendering,
and optional astronomical markers.

Problem solved
--------------
Two diverging versions of ``FITSPlotter`` coexisted simultaneously:
``fits_plotter.py`` (original) and ``fits_plotter_updated.py`` (extended
with delta-coordinate mode and interactive prompts). The classes were
never synchronized, creating a risk of feature fragmentation.
This module provides the single canonical implementation unifying all
features from both versions, plus configurable PA angles.

Author
------
Jennifer Grisales
https://github.com/jennifergc/myastroutilities
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse
from reproject import reproject_interp

from myastroutilities.structures import (
    UC1,
    get_moment_config,
    MomentConfig,
)

# Default position angles drawn when show_pa_lines=True
_DEFAULT_PA_ANGLES: tuple[int, ...] = (126, 216)

# Reference source coordinates (UC1 in M17 SW)
_UC1_RA  = "18h20m24.821s"
_UC1_DEC = "-16d11m35.02s"


class FITSPlotter:
    """
    Displays a FITS image with optional contour overlay, beam ellipses,
    source markers, and position-angle lines.

    Supports two rendering modes:

    - **WCS mode** (default): axes are RA/Dec using Astropy's WCS projection.
    - **Delta-coordinate mode** (``use_delta_coords=True``): axes are
      ΔRA/ΔDec in arcseconds relative to the reference source (UC1 by default).

    Problem solved: two near-duplicate class implementations
    (``fits_plotter.py`` and ``fits_plotter_updated.py``) coexisted in the
    ``fits_plotting_tool`` repository. This class unifies them and adds
    configurable PA angles.

    Parameters
    ----------
    image_fits : str
        Path to the base FITS image.
    contour_fits : str, optional
        Path to a FITS file used for contour overlay. Its data is reprojected
        onto the base WCS grid before plotting.
    sigma : float, optional
        Reserved for future contrast scaling. Defaults to 3e-3.
    moment : str, optional
        Moment type: ``'m0'``, ``'m1'``, ``'m2'``, or ``'continuo'``.
        Determines the colormap, colorbar label, and overlay colors.
    region_label : str, optional
        Text label displayed in the upper-right corner of the image.
    use_delta_coords : bool, optional
        If ``True``, renders axes in ΔRA/ΔDec (arcsec) relative to
        ``uc1_coord``. Defaults to ``False``.
    show_uc1 : bool, optional
        If ``True``, draws a star marker and label at ``uc1_coord``.
        Defaults to ``True``.
    show_pa_lines : bool, optional
        If ``True``, draws dashed lines at the angles specified in
        ``pa_angles``. Defaults to ``True``.
    pa_angles : tuple of int, optional
        Position angles to draw (degrees). Defaults to ``(126, 216)``.
    pa_length_pix : float, optional
        Length of the PA lines in pixels (WCS mode) or arcseconds
        (delta-coordinate mode). Defaults to 30.0.
    uc1_coord : SkyCoord, optional
        Reference coordinate used for the star marker and delta-coordinate
        origin. Defaults to the :data:`~myastroutilities.structures.UC1`
        constant (M17 SW / UC1).
    ask : bool, optional
        If ``True``, queries the user interactively when :meth:`plot` is
        called. Set to ``False`` for scripted or notebook use.
        Defaults to ``True``.

    Examples
    --------
    Programmatic use (no interactive prompts):

    >>> plotter = FITSPlotter(
    ...     "moment0.fits",
    ...     contour_fits="continuum.fits",
    ...     moment="m0",
    ...     show_uc1=True,
    ...     ask=False,
    ... )
    >>> plotter.plot(title="M17 SW — SiO Moment 0", save_as="m17_m0.png")

    Override the reference source:

    >>> from astropy.coordinates import SkyCoord
    >>> my_source = SkyCoord(ra="18h20m27s", dec="-16d12m00s", frame="icrs")
    >>> plotter = FITSPlotter("image.fits", uc1_coord=my_source, ask=False)
    """

    def __init__(
        self,
        image_fits: str,
        contour_fits: Optional[str] = None,
        sigma: float = 3e-3,
        moment: Optional[str] = None,
        region_label: Optional[str] = None,
        use_delta_coords: bool = False,
        show_uc1: bool = True,
        show_pa_lines: bool = True,
        pa_angles: tuple = _DEFAULT_PA_ANGLES,
        pa_length_pix: float = 30.0,
        uc1_coord: Optional[SkyCoord] = None,
        ask: bool = True,
    ):
        self.image_fits = image_fits
        self.contour_fits = contour_fits
        self.sigma = sigma
        self.moment = moment
        self.region_label = region_label
        self.use_delta_coords = use_delta_coords
        self.show_uc1 = show_uc1
        self.show_pa_lines = show_pa_lines
        self.pa_angles = pa_angles
        self.pa_length_pix = pa_length_pix
        self.ask = ask

        # Visual configuration derived from the moment type
        self._cfg: MomentConfig = get_moment_config(moment)
        self.colorbar_label = self._cfg.label

        # Load base image
        self.hdul_base = fits.open(self.image_fits)
        self.data_base = self.hdul_base[0].data.squeeze()
        self.wcs_base = WCS(self.hdul_base[0].header, naxis=2)
        self.pixel_scale = (
            abs(self.hdul_base[0].header.get("CDELT1", 1.0 / 3600.0)) * 3600.0
        )  # arcsec/pixel
        self.beam_base = self._get_beam_params(self.hdul_base[0].header)

        # Load and reproject contours (if provided)
        if self.contour_fits:
            self.hdul_contour = fits.open(self.contour_fits)
            self.data_contour = self.hdul_contour[0].data.squeeze()
            self.wcs_contour = WCS(self.hdul_contour[0].header, naxis=2)
            shape_out = (self.data_base.shape[-2], self.data_base.shape[-1])
            self.reprojected_contour, _ = reproject_interp(
                (self.data_contour, self.wcs_contour),
                self.wcs_base,
                shape_out=shape_out,
            )
            self.beam_contour = self._get_beam_params(self.hdul_contour[0].header)
        else:
            self.reprojected_contour = None
            self.beam_contour = None

        # Reference coordinate (defaults to UC1)
        self.uc1_coord = uc1_coord or SkyCoord(
            ra=_UC1_RA, dec=_UC1_DEC, frame="icrs"
        )

    # ── Internal utilities ───────────────────────

    @staticmethod
    def _get_beam_params(header) -> Optional[dict]:
        """
        Extracts BMAJ, BMIN (arcsec), and BPA (degrees) from a FITS header.
        Returns ``None`` when any keyword is absent.
        """
        try:
            return {
                "bmaj": header["BMAJ"] * 3600.0,
                "bmin": header["BMIN"] * 3600.0,
                "bpa":  header["BPA"],
            }
        except KeyError:
            return None

    def _arcsec_offsets_extent(self) -> tuple:
        """
        Computes the image extent in ΔRA/ΔDec (arcsec) relative to ``uc1_coord``.

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            Spatial extent in arcseconds for use as the ``extent`` argument
            of ``imshow``.
        (x_uc1, y_uc1) : tuple of float
            Pixel coordinates of ``uc1_coord`` in the base WCS.
        """
        ny, nx = self.data_base.shape[-2], self.data_base.shape[-1]
        x_uc1, y_uc1 = self.wcs_base.world_to_pixel(self.uc1_coord)

        x0_world = self.wcs_base.pixel_to_world(0, y_uc1)
        x1_world = self.wcs_base.pixel_to_world(nx - 1, y_uc1)
        y0_world = self.wcs_base.pixel_to_world(x_uc1, 0)
        y1_world = self.wcs_base.pixel_to_world(x_uc1, ny - 1)

        dec_rad = np.deg2rad(self.uc1_coord.dec.deg)
        dra0 = (x0_world.ra.deg - self.uc1_coord.ra.deg) * np.cos(dec_rad) * 3600.0
        dra1 = (x1_world.ra.deg - self.uc1_coord.ra.deg) * np.cos(dec_rad) * 3600.0
        ddec0 = (y0_world.dec.deg - self.uc1_coord.dec.deg) * 3600.0
        ddec1 = (y1_world.dec.deg - self.uc1_coord.dec.deg) * 3600.0

        xmin, xmax = np.sort([dra0, dra1])
        ymin, ymax = np.sort([ddec0, ddec1])
        return xmin, xmax, ymin, ymax, (x_uc1, y_uc1)

    # ── Beam drawing ─────────────────────────────

    def _draw_beam_pixels(self, ax, beam: dict, facecolor: str, edgecolor: str):
        """
        Draws a synthesized beam ellipse in the lower-left corner of the plot,
        using pixel coordinates (WCS-projection axes).
        """
        if not beam:
            return
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        bx = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        by = ylim[0] + 0.05 * (ylim[1] - ylim[0])
        ax.add_patch(Ellipse(
            (bx, by),
            width=beam["bmin"] / self.pixel_scale,
            height=beam["bmaj"] / self.pixel_scale,
            angle=beam["bpa"],
            edgecolor=edgecolor, facecolor=facecolor, alpha=0.5, lw=1.5,
        ))

    def _draw_beam_arcsec(self, ax, beam: dict, facecolor: str, edgecolor: str,
                          xmin: float, xmax: float, ymin: float, ymax: float):
        """
        Draws a synthesized beam ellipse in the lower-left corner of the plot,
        using arcsecond coordinates (delta-coordinate mode).
        """
        if not beam:
            return
        bx = xmin + 0.05 * (xmax - xmin)
        by = ymin + 0.05 * (ymax - ymin)
        ax.add_patch(Ellipse(
            (bx, by),
            width=beam["bmin"],
            height=beam["bmaj"],
            angle=beam["bpa"],
            edgecolor=edgecolor, facecolor=facecolor, alpha=0.5, lw=1.5,
        ))

    # ── PA line drawing ──────────────────────────

    def _draw_pa_lines_pixels(self, ax, x_pix: float, y_pix: float, length: float):
        """
        Draws dashed position-angle lines through (x_pix, y_pix) in pixel space.
        """
        for pa in self.pa_angles:
            theta = np.deg2rad(pa)
            dx = length * np.sin(theta)
            dy = length * np.cos(theta)
            ax.plot([x_pix - dx, x_pix + dx], [y_pix - dy, y_pix + dy],
                    transform=ax.get_transform("pixel"),
                    color="gray", linestyle="--", linewidth=1)
            ax.text(x_pix - 0.8 * dx, y_pix - 0.8 * dy, f"PA={pa}°",
                    transform=ax.get_transform("pixel"),
                    color="gray", fontsize=12, rotation_mode="anchor")

    def _draw_pa_lines_arcsec(self, ax, length_arcsec: float):
        """
        Draws dashed position-angle lines through the origin (UC1) in arcsec space.
        """
        for pa in self.pa_angles:
            theta = np.deg2rad(pa)
            dx = length_arcsec * np.sin(theta)
            dy = length_arcsec * np.cos(theta)
            ax.plot([-dx, dx], [-dy, dy],
                    color="gray", linestyle="--", linewidth=1)
            ax.text(-0.8 * dx, -0.8 * dy, f"PA={pa}°",
                    color="gray", fontsize=12, rotation_mode="anchor")

    # ── Interactive option prompts ───────────────

    def _ask_options(self):
        """
        Queries the user interactively for rendering options.
        Updates ``use_delta_coords``, ``show_uc1``, and ``show_pa_lines``.
        """
        options = [
            ("¿Mostrar ejes en ΔRA/ΔDec relativos a UC1? [s/N]: ",
             "use_delta_coords", "s"),
            ("¿Mostrar la estrellita de UC1? [S/n]: ",
             "show_uc1", "n"),
            ("¿Mostrar las líneas PA? [S/n]: ",
             "show_pa_lines", "n"),
        ]
        for question, attr, off_answer in options:
            try:
                resp = input(question).strip().lower()
                current = getattr(self, attr)
                setattr(self, attr, not current if resp == off_answer else current
                        if resp == "" else (resp != off_answer))
            except Exception:
                pass

    # ── Main rendering ───────────────────────────

    def plot(self, save_as: Optional[str] = None, title: str = ""):
        """
        Renders the FITS image with all configured overlays.

        Parameters
        ----------
        save_as : str, optional
            File path for saving the figure (PNG, PDF, etc.).
            Resolution is set to 300 dpi.
        title : str, optional
            Figure title.
        """
        if self.ask:
            self._ask_options()

        cmap_base = self._cfg.colormap
        contour_color = self._cfg.contour_color
        star_color = self._cfg.star_color

        if not self.use_delta_coords:
            self._render_wcs(title, save_as, cmap_base, contour_color, star_color)
        else:
            self._render_delta(title, save_as, cmap_base, contour_color, star_color)

    def _render_wcs(self, title, save_as, cmap_base, contour_color, star_color):
        """Renders the image with standard WCS (RA/Dec) axes."""
        fig, ax = plt.subplots(figsize=(10, 8),
                               subplot_kw={"projection": self.wcs_base})

        im = ax.imshow(self.data_base, origin="lower", cmap=cmap_base)

        if self.reprojected_contour is not None:
            self._draw_contours(ax, self.reprojected_contour,
                                self.data_contour, contour_color)

        self._draw_beam_pixels(ax, self.beam_base, "gray", "black")
        if self.beam_contour:
            self._draw_beam_pixels(ax, self.beam_contour, "white", "gray")

        if self.region_label:
            ax.text(0.95, 0.95, self.region_label, transform=ax.transAxes,
                    fontsize=14, color="white", ha="right", va="top",
                    bbox=dict(facecolor="black", alpha=0.5))

        x_pix, y_pix = self.wcs_base.world_to_pixel(self.uc1_coord)

        if self.show_uc1:
            ax.scatter(x_pix, y_pix, facecolors="none", edgecolors=star_color,
                       marker="*", s=250, linewidths=1.5, zorder=10,
                       transform=ax.get_transform("pixel"))
            ax.annotate("UC1", (x_pix + 5, y_pix + 5), color=star_color,
                        fontsize=12, weight="bold", zorder=11,
                        transform=ax.get_transform("pixel"))

        if self.show_pa_lines:
            self._draw_pa_lines_pixels(ax, x_pix, y_pix, self.pa_length_pix)

        ax.set_xlabel("Right Ascension (RA)")
        ax.set_ylabel("Declination (Dec)")
        plt.colorbar(im, ax=ax, pad=0.05, label=self.colorbar_label)
        plt.title(title)
        self._finalize(fig, save_as)

    def _render_delta(self, title, save_as, cmap_base, contour_color, star_color):
        """Renders the image with ΔRA/ΔDec (arcsec) axes relative to UC1."""
        fig, ax = plt.subplots(figsize=(10, 8))
        xmin, xmax, ymin, ymax, _ = self._arcsec_offsets_extent()

        im = ax.imshow(self.data_base, origin="lower", cmap=cmap_base,
                       extent=(xmin, xmax, ymin, ymax), aspect="equal")

        if self.reprojected_contour is not None:
            self._draw_contours(ax, self.reprojected_contour,
                                self.data_contour, contour_color,
                                extent=(xmin, xmax, ymin, ymax))

        self._draw_beam_arcsec(ax, self.beam_base, "gray", "black",
                               xmin, xmax, ymin, ymax)
        if self.beam_contour:
            self._draw_beam_arcsec(ax, self.beam_contour, "white", "gray",
                                   xmin, xmax, ymin, ymax)

        if self.region_label:
            ax.text(0.95, 0.95, self.region_label, transform=ax.transAxes,
                    fontsize=14, color="white", ha="right", va="top",
                    bbox=dict(facecolor="black", alpha=0.5))

        if self.show_uc1:
            ax.scatter(0.0, 0.0, facecolors="none", edgecolors=star_color,
                       marker="*", s=250, linewidths=1.5, zorder=10)
            ax.annotate("UC1", (3.0, 3.0), color=star_color,
                        fontsize=12, weight="bold", zorder=11)

        if self.show_pa_lines:
            self._draw_pa_lines_arcsec(ax, self.pa_length_pix)

        ax.set_xlabel("ΔRA (arcsec)")
        ax.set_ylabel("ΔDec (arcsec)")
        plt.colorbar(im, ax=ax, pad=0.05, label=self.colorbar_label)
        plt.title(title)
        self._finalize(fig, save_as)

    @staticmethod
    def _draw_contours(ax, data_reprojected, data_original, color,
                       extent=None):
        """
        Draws contour lines from reprojected data onto the current axes.
        Skips drawing when the data range is degenerate.
        """
        vmin = np.nanmin(data_original)
        vmax = np.nanmax(data_original)
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
            return
        levels = np.linspace(vmin, vmax, 7)
        kwargs = dict(levels=levels, colors=color, linewidths=1, alpha=0.8)
        if extent is not None:
            kwargs["extent"] = extent
        ax.contour(data_reprojected, **kwargs)

    @staticmethod
    def _finalize(fig, save_as: Optional[str]):
        """Saves the figure to disk (if requested) and displays it."""
        if save_as:
            fig.savefig(save_as, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_as}")
        plt.show()

    # ── Backward-compatibility aliases ───────────

    def get_beam_params(self, header) -> Optional[dict]:
        """Alias for :meth:`_get_beam_params`. Preserved for compatibility."""
        return self._get_beam_params(header)

    def plot_beam(self, ax, beam_params, facecolor, edgecolor):
        """Alias for :meth:`_draw_beam_pixels`. Preserved for compatibility."""
        self._draw_beam_pixels(ax, beam_params, facecolor, edgecolor)

    def plot_beam_pixels(self, ax, beam_params, facecolor, edgecolor):
        """Alias for :meth:`_draw_beam_pixels`. Preserved for compatibility."""
        self._draw_beam_pixels(ax, beam_params, facecolor, edgecolor)

    def plot_beam_arcsec(self, ax, beam_params, facecolor, edgecolor,
                         xmin, xmax, ymin, ymax):
        """Alias for :meth:`_draw_beam_arcsec`. Preserved for compatibility."""
        self._draw_beam_arcsec(ax, beam_params, facecolor, edgecolor,
                               xmin, xmax, ymin, ymax)


# ─────────────────────────────────────────────
#   CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="FITSPlotter — FITS image visualizer with contour overlays."
    )
    p.add_argument("--image",     required=True,       help="Base FITS image path")
    p.add_argument("--contours",  default=None,         help="Contour FITS path (optional)")
    p.add_argument("--moment",    default=None,         help="m0, m1, m2, or continuo")
    p.add_argument("--label",     default=None,         help="Region label")
    p.add_argument("--delta",     action="store_true",  help="Use ΔRA/ΔDec axes")
    p.add_argument("--no-ask",    action="store_true",  help="Skip interactive prompts")
    p.add_argument("--hide-uc1",  action="store_true",  help="Hide UC1 marker")
    p.add_argument("--hide-pa",   action="store_true",  help="Hide PA lines")
    p.add_argument("--pa-angles", nargs="+", type=int,
                   default=list(_DEFAULT_PA_ANGLES),    help="PA angles to draw (degrees)")
    p.add_argument("--save",      default=None,         help="Output figure path")
    args = p.parse_args()

    plotter = FITSPlotter(
        image_fits=args.image,
        contour_fits=args.contours,
        moment=args.moment,
        region_label=args.label,
        use_delta_coords=args.delta,
        show_uc1=(not args.hide_uc1),
        show_pa_lines=(not args.hide_pa),
        pa_angles=tuple(args.pa_angles),
        ask=(not args.no_ask),
    )
    plotter.plot(save_as=args.save, title=args.label or "")
