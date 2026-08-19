#!/usr/bin/env python3
"""
dump_fits_headers.py

Vuelca el header completo de uno o mas mapas FITS a archivos .txt, y genera
ademas un resumen con las claves relevantes para reconstruir las condiciones
de observacion e imagenado (configuracion VLA, ponderacion, uvrange, haz).

Uso:
    python dump_fits_headers.py mapa_X.fits mapa_K.fits
    python dump_fits_headers.py *.fits -o headers/

Salida:
    <outdir>/<nombre>_header.txt     header completo, HDU por HDU
    <outdir>/header_summary.txt      tabla resumen de todos los mapas
"""

import argparse
import os
import sys
from astropy.io import fits

# Claves que interesan para la tabla de observaciones del paper.
SUMMARY_KEYS = [
    "OBJECT", "TELESCOP", "INSTRUME", "OBSERVER", "ORIGIN",
    "DATE-OBS", "DATE", "MJD-OBS", "OBSRA", "OBSDEC",
    "BUNIT", "BMAJ", "BMIN", "BPA",
    "NAXIS1", "NAXIS2", "CDELT1", "CDELT2",
    "CTYPE3", "CRVAL3", "CDELT3", "RESTFRQ", "RESTFREQ",
    "SPECSYS", "EQUINOX", "RADESYS",
]

# Subcadenas que se buscan dentro de las tarjetas HISTORY (CASA tclean / AIPS).
HISTORY_FLAGS = [
    "weighting", "robust", "uvtaper", "uvrange", "deconvolver",
    "gridder", "cell", "imsize", "specmode", "nterms", "scales",
    "pbcor", "restoringbeam", "niter", "threshold",
    "IMAGR", "UVWTFN", "ROBUST", "UVRANG",
]


def beam_from_casambm(hdul):
    """Recupera el haz si esta en la extension binaria CASAMBM (multi-beam)."""
    for hdu in hdul:
        name = (hdu.name or "").upper()
        if name in ("CASAMBM", "BEAMS") and hasattr(hdu, "data") and hdu.data is not None:
            cols = [c.upper() for c in hdu.data.columns.names]
            if {"BMAJ", "BMIN", "BPA"}.issubset(set(cols)):
                d = hdu.data
                return {
                    "BMAJ_median": float(
                        sorted(d["BMAJ"])[len(d["BMAJ"]) // 2]),
                    "BMIN_median": float(
                        sorted(d["BMIN"])[len(d["BMIN"]) // 2]),
                    "BPA_median": float(
                        sorted(d["BPA"])[len(d["BPA"]) // 2]),
                    "n_planes": len(d),
                    "unit": hdu.header.get("TUNIT1", "?"),
                }
    return None


def dump_one(path, outdir):
    base = os.path.splitext(os.path.basename(path))[0]
    out_txt = os.path.join(outdir, f"{base}_header.txt")

    with fits.open(path, memmap=False) as hdul:
        lines = []
        lines.append("=" * 78)
        lines.append(f"ARCHIVO: {os.path.abspath(path)}")
        lines.append(f"TAMANO : {os.path.getsize(path) / 1e6:.2f} MB")
        lines.append("=" * 78)
        lines.append("")
        lines.append("--- ESTRUCTURA (hdul.info) ---")
        info = hdul.info(output=False)
        for row in info:
            lines.append("  " + "  ".join(str(x) for x in row))
        lines.append("")

        for i, hdu in enumerate(hdul):
            lines.append("-" * 78)
            lines.append(f"--- HDU {i}  (name={hdu.name!r}, "
                         f"class={type(hdu).__name__}) ---")
            lines.append("-" * 78)
            # repr(header) preserva HISTORY, COMMENT y el formato de 80 col.
            lines.append(repr(hdu.header))
            lines.append("")

        with open(out_txt, "w") as f:
            f.write("\n".join(lines))

        # --- resumen ---
        hdr = hdul[0].header
        summary = {k: hdr.get(k, None) for k in SUMMARY_KEYS}
        summary = {k: v for k, v in summary.items() if v is not None}

        mb = beam_from_casambm(hdul)
        if mb is not None:
            summary["CASAMBM"] = mb

        hist = []
        if "HISTORY" in hdr:
            hist = [str(c) for c in hdr["HISTORY"]]
        flagged = [h for h in hist
                   if any(k.lower() in h.lower() for k in HISTORY_FLAGS)]

    return out_txt, summary, hist, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="mapas FITS")
    ap.add_argument("-o", "--outdir", default="headers",
                    help="directorio de salida (default: headers/)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_lines = []

    for path in args.files:
        if not os.path.isfile(path):
            print(f"[AVISO] no existe o no es archivo: {path}", file=sys.stderr)
            continue
        try:
            out_txt, summary, hist, flagged = dump_one(path, args.outdir)
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            continue

        print(f"[OK] {path} -> {out_txt}")

        summary_lines.append("=" * 78)
        summary_lines.append(f"ARCHIVO: {os.path.abspath(path)}")
        summary_lines.append("=" * 78)
        for k, v in summary.items():
            summary_lines.append(f"  {k:<12s} = {v}")

        # Haz en arcsec si BMAJ/BMIN estan en grados (convencion FITS).
        if "BMAJ" in summary and "BMIN" in summary:
            try:
                summary_lines.append(
                    f"  {'BEAM[arcsec]':<12s} = "
                    f"{summary['BMAJ'] * 3600:.4f} x "
                    f"{summary['BMIN'] * 3600:.4f}")
            except (TypeError, ValueError):
                pass
        if "CRVAL3" in summary:
            try:
                summary_lines.append(
                    f"  {'FREQ[GHz]':<12s} = {summary['CRVAL3'] / 1e9:.6f}")
            except (TypeError, ValueError):
                pass

        summary_lines.append("")
        summary_lines.append(f"  Tarjetas HISTORY totales: {len(hist)}")
        summary_lines.append("  HISTORY con parametros de imagenado:")
        if flagged:
            for h in flagged:
                summary_lines.append(f"    | {h}")
        else:
            summary_lines.append("    (ninguna: el header no conserva la "
                                 "llamada de imagenado)")
        summary_lines.append("")

    out_summary = os.path.join(args.outdir, "header_summary.txt")
    with open(out_summary, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"[OK] resumen -> {out_summary}")


if __name__ == "__main__":
    main()
