#!/usr/bin/env python
"""Dataset download helpers for xsretrieval.

Fetches the small **EuroSAT** dataset (the day-1 same-modal optical/MS sanity
set) and prints actionable download instructions + URLs for the larger
co-registered multi-modal datasets (SEN12MS, BigEarthNet-MM, QXS-SAROPT, …),
summarised from ``research/01_datasets.md``.

Usage
-----
    python scripts/download_data.py eurosat --root data/eurosat       # RGB (~90 MB)
    python scripts/download_data.py eurosat --root data/eurosat --bands all  # 13-band (~2 GB)
    python scripts/download_data.py instructions                      # print the rest

EuroSAT download is stdlib-only (urllib + zipfile). The big multi-modal sets
require accounts / large storage and are only described, not auto-downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

# Mirrors xsretrieval.data.datasets.DATASET_URLS (kept here so the script is
# self-contained and runnable before the package is installed).
EUROSAT_RGB_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
EUROSAT_MS_URL = "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip"

DATASETS = {
    "SEN12MS": {
        "what": "SAR(VV/VH)+MS(13b)+RGB(derived)+IGBP landcover, pixel-aligned",
        "size": "~430 GB  (180,662 triplets, 10 m, 256x256)",
        "license": "CC-BY (TUM mediaTUM)",
        "url": "https://mediatum.ub.tum.de/1474000",
        "note": "Primary tri-modal set. Also via torchgeo `SEN12MS`. Use DFC2020 "
        "for high-res labelled test split.",
    },
    "DFC2020": {
        "what": "Curated SEN12MS subset with high-res (10 m) labels (held-out test)",
        "size": "~25 GB",
        "license": "research (IEEE GRSS)",
        "url": "https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest",
        "note": "Trustworthy labels for the official cross-modal F1 eval split.",
    },
    "BigEarthNet-MM": {
        "what": "SAR(VV/VH)+MS, 19-class CLC2018 multi-label, pixel-aligned",
        "size": "~120 GB  (590,326 pairs, 120x120)",
        "license": "CDLA-Permissive 1.0 (commercial-friendly)",
        "url": "https://bigearth.net/",
        "note": "v2 (reBEN) on Zenodo: https://zenodo.org/records/10891137 . "
        "torchgeo `BigEarthNet`. Multi-label => use >=1 shared label as relevance.",
    },
    "QXS-SAROPT": {
        "what": "High-res (1 m) SAR<->optical co-registered pairs (no labels)",
        "size": "~2 GB  (20,000 pairs, 256x256)",
        "license": "research (cite paper)",
        "url": "https://github.com/yaoxu008/QXS-SAROPT",
        "note": "VHR SAR-optical generalization; relevance = geographic pair identity.",
    },
    "So2Sat-LCZ42": {
        "what": "SAR+MS with Local Climate Zone labels (17 classes)",
        "size": "~55 GB",
        "license": "CC-BY (TUM)",
        "url": "https://mediatum.ub.tum.de/1454690",
        "note": "Urban-focused LCZ classes; clean labels for cross-modal F1.",
    },
}


def _download(url: str, dest_zip: str) -> None:
    """Download *url* to *dest_zip* with a simple progress bar (stdlib)."""

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        mb = done / 1e6
        tot_mb = total_size / 1e6
        sys.stdout.write(f"\r  downloading: {pct:5.1f}%  ({mb:.1f}/{tot_mb:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_zip, _hook)
    sys.stdout.write("\n")


def fetch_eurosat(root: str, bands: str = "rgb") -> int:
    """Download + extract EuroSAT into *root*. ``bands`` is ``"rgb"`` or ``"all"``."""
    os.makedirs(root, exist_ok=True)
    url = EUROSAT_MS_URL if bands == "all" else EUROSAT_RGB_URL
    dest_zip = os.path.join(root, os.path.basename(url))
    print(f"[eurosat] target: {root}  bands={bands}")
    print(f"[eurosat] source: {url}")
    if os.path.exists(dest_zip):
        print(f"[eurosat] archive already present: {dest_zip}")
    else:
        try:
            _download(url, dest_zip)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[eurosat] download failed: {exc}", file=sys.stderr)
            print(f"[eurosat] download manually from {url} and unzip into {root}")
            return 1
    print(f"[eurosat] extracting {dest_zip} ...")
    try:
        with zipfile.ZipFile(dest_zip) as zf:
            zf.extractall(root)
    except Exception as exc:  # pragma: no cover - file dependent
        print(f"[eurosat] extract failed: {exc}", file=sys.stderr)
        return 1
    print(f"[eurosat] done. Point a config at it:\n"
          f"  data.dataset: eurosat\n  data.root: {root}")
    return 0


def print_instructions() -> int:
    """Print download instructions + URLs for the larger multi-modal datasets."""
    print("=" * 78)
    print("Multi-modal satellite datasets (from research/01_datasets.md)")
    print("=" * 78)
    print("EuroSAT (auto-downloadable here): small optical/MS, 10 classes, 64x64.")
    print(f"  RGB  (~90 MB): {EUROSAT_RGB_URL}")
    print(f"  13-b (~2 GB) : {EUROSAT_MS_URL}")
    print("  -> python scripts/download_data.py eurosat --root data/eurosat\n")
    for name, info in DATASETS.items():
        print(f"{name}")
        print(f"  modalities : {info['what']}")
        print(f"  size       : {info['size']}")
        print(f"  license    : {info['license']}")
        print(f"  url        : {info['url']}")
        print(f"  note       : {info['note']}")
        print()
    print("Recommended convergent strategy: SEN12MS (primary tri-modal) + DFC2020 "
          "(labelled test) + EuroSAT (day-1 harness) + QXS-SAROPT (VHR SAR-optical).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="xsretrieval dataset helpers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_euro = sub.add_parser("eurosat", help="download + extract EuroSAT")
    p_euro.add_argument("--root", default="data/eurosat", help="target directory")
    p_euro.add_argument("--bands", choices=["rgb", "all"], default="rgb",
                        help="rgb (~90 MB) or all=13-band (~2 GB)")

    sub.add_parser("instructions", help="print download guide for big datasets")

    args = parser.parse_args(argv)
    if args.command == "eurosat":
        return fetch_eurosat(args.root, args.bands)
    return print_instructions()


if __name__ == "__main__":
    raise SystemExit(main())
