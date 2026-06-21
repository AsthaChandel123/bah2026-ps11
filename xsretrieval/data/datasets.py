"""Dataset adapters that yield :class:`Sample` lists for cross-modal retrieval.

Every adapter is **lazy about heavy dependencies** (PIL / rasterio are imported
inside methods) and **degrades gracefully** when data is not present on disk: it
raises a clear :class:`FileNotFoundError` that includes the official download URL
(from ``research/01_datasets.md``) rather than crashing obscurely.

Adapters provided
-----------------
* :class:`FolderDataset` -- generic image-folder reader (one sub-dir per class),
  modality-tagged. Covers EuroSAT-RGB, NWPU-RESISC45, PatternNet, UC Merced, etc.
* :class:`EuroSATDataset` -- EuroSAT, RGB *or* all-13-bands variant.
* :class:`SEN12MSDataset` -- co-registered Sentinel-1 (SAR) + Sentinel-2 (MS) +
  derived RGB + IGBP land-cover label, reading the standard SEN12MS layout.
* :class:`MultiModalDataset` -- groups loaded :class:`Sample` objects by
  ``location_id`` across modalities and builds query/gallery splits for both
  same-modal and cross-modal evaluation.
* :func:`make_query_gallery_split` -- stratified query/gallery splitter.
* :func:`get_dataset` -- small name -> adapter registry.

Design choices
--------------
Adapters return ``list[Sample]`` of *raw* images (no preprocessing applied), so
the caller controls resizing/normalisation via
:func:`xsretrieval.data.preprocessing.preprocess`. Optional ``preprocess=True``
applies the modality-appropriate pipeline eagerly.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .modalities import Modality, Sample, to_rgb

__all__ = [
    "DownloadInfo",
    "DATASET_URLS",
    "FolderDataset",
    "EuroSATDataset",
    "SEN12MSDataset",
    "MultiModalDataset",
    "make_query_gallery_split",
    "get_dataset",
    "load_image_any",
]


# Official download URLs surfaced in error messages (from research/01_datasets.md).
DATASET_URLS: dict[str, str] = {
    "eurosat_rgb": "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
    "eurosat_ms": "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip",
    "sen12ms": "https://mediatum.ub.tum.de/1474000",
    "bigearthnet": "https://bigearth.net/",
    "so2sat": "https://mediatum.ub.tum.de/1454690",
    "patternnet": "https://sites.google.com/view/zhouwx/dataset",
    "resisc45": "https://gcheng-nwpu.github.io/",
    "aid": "https://captain-whu.github.io/AID/",
    "ucmerced": "http://weegee.vision.ucmerced.edu/datasets/landuse.html",
}

# Common raster file extensions handled by the loaders.
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
_GEO_EXTS = (".tif", ".tiff")


@dataclass(frozen=True)
class DownloadInfo:
    """Where to obtain a dataset (used in :class:`FileNotFoundError` messages)."""

    name: str
    url: str
    note: str = ""


def _require_dir(path: str, info: DownloadInfo) -> None:
    """Raise an actionable :class:`FileNotFoundError` if ``path`` is absent."""
    if not os.path.isdir(path):
        msg = (
            f"[{info.name}] expected data directory not found:\n  {path!r}\n"
            f"Download it from: {info.url}"
        )
        if info.note:
            msg += f"\n{info.note}"
        raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Image loading helpers (lazy).
# ---------------------------------------------------------------------------
def load_image_any(path: str) -> np.ndarray:
    """Load a raster file into a ``(C, H, W)`` float32 array.

    Dispatches on extension: GeoTIFF (``.tif``/``.tiff``) via ``rasterio`` (lazy;
    preserves all bands), everything else via PIL. Both dependencies are imported
    inside this function so the module loads without them.

    Parameters
    ----------
    path:
        Path to an image / raster file.

    Returns
    -------
    np.ndarray
        ``(C, H, W)`` float32 array (single-band images become ``C=1``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If the required backend (rasterio for GeoTIFF, PIL otherwise) is missing,
        with an install hint.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image file not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext in _GEO_EXTS:
        try:
            import rasterio  # lazy, heavy/optional
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Reading GeoTIFF requires 'rasterio'. Install with "
                "`pip install rasterio`."
            ) from exc
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
        return arr.astype(np.float32)
    # Raster via PIL.
    try:
        from PIL import Image  # lazy
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "Reading images requires 'pillow'. Install with `pip install pillow`."
        ) from exc
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def _scan_class_folders(root: str) -> list[tuple[str, str]]:
    """Return ``(filepath, class_name)`` for an ImageFolder-style ``root``.

    Layout: ``root/<class_name>/<image files>``. Only raster files are kept; the
    list is sorted for deterministic ordering.
    """
    items: list[tuple[str, str]] = []
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(_IMG_EXTS + _GEO_EXTS):
                items.append((os.path.join(cdir, fn), cls))
    return items


# ---------------------------------------------------------------------------
# FolderDataset.
# ---------------------------------------------------------------------------
class FolderDataset:
    """Generic image-folder dataset (one sub-directory per class), modality-tagged.

    Reads an ImageFolder-style hierarchy ``root/<class>/<image>`` and emits one
    :class:`Sample` per image, all tagged with a single ``modality``. Suitable
    for the many optical-RGB scene-classification sets (EuroSAT-RGB, NWPU-
    RESISC45, AID, PatternNet, UC Merced) and, with ``modality=MULTISPECTRAL``,
    for band-stacked GeoTIFF folders.

    The class is iterable (``for s in ds``) and indexable (``ds[i]``); images are
    loaded lazily on access so large folders do not exhaust memory.

    Parameters
    ----------
    root:
        Dataset root containing per-class sub-directories.
    modality:
        Modality to tag every sample with.
    label_from:
        How to derive the integer label. ``"dirname"`` (default) maps each class
        directory name to a stable integer id; ``"none"`` sets ``label=-1``.
    size:
        If not ``None``, eagerly resize/normalise via
        :func:`~xsretrieval.data.preprocessing.preprocess` on access.
    preprocess:
        Apply modality preprocessing on access (requires ``size``).
    transform:
        Optional callable applied to the raw ``(C, H, W)`` image after loading.
    location_from:
        Optional callable mapping a file path to a ``location_id`` (enables
        cross-modal grouping when filenames encode geography).
    """

    def __init__(
        self,
        root: str,
        modality: Modality = Modality.OPTICAL_RGB,
        label_from: str = "dirname",
        *,
        size: Optional[int] = None,
        preprocess: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        location_from: Optional[Callable[[str], str]] = None,
    ) -> None:
        info = DownloadInfo("FolderDataset", DATASET_URLS.get("resisc45", root))
        _require_dir(root, info)
        self.root = root
        self.modality = Modality(modality)
        self.label_from = label_from
        self.size = size
        self.preprocess = preprocess
        self.transform = transform
        self.location_from = location_from

        self._items = _scan_class_folders(root)
        if not self._items:
            raise FileNotFoundError(
                f"[FolderDataset] no images found under {root!r}. Expected "
                f"layout root/<class>/<image>.{_IMG_EXTS + _GEO_EXTS}"
            )
        classes = sorted({cls for _, cls in self._items})
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.classes = classes

    def __len__(self) -> int:
        return len(self._items)

    def _make_sample(self, idx: int) -> Sample:
        path, cls = self._items[idx]
        img = load_image_any(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.preprocess and self.size is not None:
            from .preprocessing import preprocess as _pp

            img = _pp(img, self.modality, size=self.size)
        elif self.size is not None:
            from .preprocessing import resize_image

            img = resize_image(img, self.size)
        label = (
            self.class_to_idx[cls] if self.label_from == "dirname" else -1
        )
        loc = self.location_from(path) if self.location_from else None
        sample_id = os.path.relpath(path, self.root)
        return Sample(
            id=sample_id,
            image=img.astype(np.float32, copy=False),
            modality=self.modality,
            label=label,
            location_id=loc,
            meta={"path": path, "class_name": cls},
        )

    def __getitem__(self, idx: int) -> Sample:
        return self._make_sample(idx)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self._make_sample(i)

    def to_list(self, limit: Optional[int] = None) -> list[Sample]:
        """Materialise (a prefix of) the dataset into a list of samples."""
        n = len(self) if limit is None else min(limit, len(self))
        return [self._make_sample(i) for i in range(n)]


# ---------------------------------------------------------------------------
# EuroSATDataset.
# ---------------------------------------------------------------------------
class EuroSATDataset:
    """EuroSAT land-use dataset (10 classes, Sentinel-2), RGB or all-13-bands.

    EuroSAT ships in two variants:

    * **RGB** -- ``.jpg`` images in ``root/<class>/`` (``variant="rgb"``,
      modality OPTICAL_RGB).
    * **All bands** -- 13-band ``.tif`` patches in ``root/<class>/``
      (``variant="ms"``, modality MULTISPECTRAL).

    A single-label scene dataset, ideal as the day-1 same-modal optical / MS
    retrieval harness (see research recommendation). For ``variant="ms"`` an
    optional derived RGB modality can also be emitted (``derive_rgb=True``),
    giving a cheap two-modality (MS + RGB) set sharing ``location_id`` per patch.

    Parameters
    ----------
    root:
        EuroSAT root with per-class sub-directories.
    variant:
        ``"rgb"`` or ``"ms"`` (all-bands).
    derive_rgb:
        MS-variant only -- additionally emit an OPTICAL_RGB sample per patch,
        derived from B4/B3/B2 via :func:`~xsretrieval.data.modalities.to_rgb`,
        sharing the patch's ``location_id`` (creates cross-modal positives).
    size, preprocess, transform:
        As in :class:`FolderDataset`.
    """

    def __init__(
        self,
        root: str,
        variant: str = "rgb",
        *,
        derive_rgb: bool = False,
        size: Optional[int] = None,
        preprocess: bool = False,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        variant = variant.lower()
        if variant not in ("rgb", "ms"):
            raise ValueError("variant must be 'rgb' or 'ms'")
        url_key = "eurosat_rgb" if variant == "rgb" else "eurosat_ms"
        info = DownloadInfo("EuroSAT", DATASET_URLS[url_key])
        _require_dir(root, info)
        self.root = root
        self.variant = variant
        self.derive_rgb = derive_rgb and variant == "ms"
        self.size = size
        self.preprocess = preprocess
        self.transform = transform
        self.modality = (
            Modality.OPTICAL_RGB if variant == "rgb" else Modality.MULTISPECTRAL
        )
        self._items = _scan_class_folders(root)
        if not self._items:
            raise FileNotFoundError(
                f"[EuroSAT] no images found under {root!r}; download from "
                f"{info.url}"
            )
        classes = sorted({cls for _, cls in self._items})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes

    def __len__(self) -> int:
        return len(self._items)

    def _emit(self, path: str, cls: str) -> list[Sample]:
        img = load_image_any(path)
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        loc = os.path.splitext(os.path.relpath(path, self.root))[0]
        out: list[Sample] = []

        def _post(arr: np.ndarray, modality: Modality) -> np.ndarray:
            if self.preprocess and self.size is not None:
                from .preprocessing import preprocess as _pp

                return _pp(arr, modality, size=self.size)
            if self.size is not None:
                from .preprocessing import resize_image

                return resize_image(arr, self.size)
            return arr.astype(np.float32, copy=False)

        out.append(
            Sample(
                id=f"{self.modality.value}:{loc}",
                image=_post(img, self.modality),
                modality=self.modality,
                label=label,
                location_id=loc,
                meta={"path": path, "class_name": cls},
            )
        )
        if self.derive_rgb and img.shape[0] >= 4:
            rgb = to_rgb(img)
            out.append(
                Sample(
                    id=f"optical_rgb:{loc}",
                    image=_post(rgb, Modality.OPTICAL_RGB),
                    modality=Modality.OPTICAL_RGB,
                    label=label,
                    location_id=loc,
                    meta={"path": path, "class_name": cls, "derived": "rgb"},
                )
            )
        return out

    def __iter__(self) -> Iterator[Sample]:
        for path, cls in self._items:
            yield from self._emit(path, cls)

    def to_list(self, limit: Optional[int] = None) -> list[Sample]:
        """Materialise samples (each patch may yield 1 or 2 samples)."""
        out: list[Sample] = []
        for path, cls in self._items:
            out.extend(self._emit(path, cls))
            if limit is not None and len(out) >= limit:
                return out[:limit]
        return out


# ---------------------------------------------------------------------------
# SEN12MSDataset.
# ---------------------------------------------------------------------------
class SEN12MSDataset:
    """SEN12MS adapter: co-registered Sentinel-1 SAR + Sentinel-2 MS + IGBP label.

    SEN12MS is the recommended *primary* dataset: 180k pixel-aligned triplets of
    Sentinel-1 dual-pol SAR (VV/VH), Sentinel-2 13-band multispectral, and a
    MODIS-derived IGBP land-cover map. From S2 an RGB modality is derived for
    free. This adapter emits, per patch, up to three co-registered samples (SAR,
    MS, optional RGB) sharing one ``location_id`` -- exactly the cross-modal
    positive structure the system trains on.

    Expected directory layout (the official SEN12MS release)::

        root/
          ROIs<id>_<season>/
            s1_<scene>/   ROIs..._s1_<scene>_p<patch>.tif   # 2 bands: VV, VH
            s2_<scene>/   ROIs..._s2_<scene>_p<patch>.tif   # 13 bands
            lc_<scene>/   ROIs..._lc_<scene>_p<patch>.tif   # land-cover (IGBP)

    The SAR / MS / LC patches for the same ``<scene>_p<patch>`` are pixel-aligned.
    Reading GeoTIFFs requires ``rasterio`` (imported lazily).

    Download: https://mediatum.ub.tum.de/1474000

    Parameters
    ----------
    root:
        SEN12MS root directory.
    season:
        Optional season filter (e.g. ``"spring"``); matches the ``<season>`` part
        of the ``ROIs..._<season>`` directory names. ``None`` = all seasons.
    modalities:
        Subset of {SAR, MULTISPECTRAL, OPTICAL_RGB} to emit. OPTICAL_RGB is
        derived from the S2 bands.
    size, preprocess:
        Optional eager resize / preprocessing.
    igbp_simplified:
        Map the 17-class IGBP scheme to the simplified 10-class DFC2020 scheme
        used for evaluation (recommended).
    max_patches:
        Cap the number of patches scanned (useful for quick prototyping).
    """

    #: IGBP 17 -> simplified 10-class mapping (the DFC2020 scheme).
    IGBP17_TO_SIMPLIFIED10: dict[int, int] = {
        1: 1, 2: 1, 3: 1, 4: 1, 5: 1,       # forests -> 1
        6: 2, 7: 2,                          # shrubland -> 2
        8: 3, 9: 3,                          # savanna -> 3
        10: 4,                               # grassland -> 4
        11: 5,                               # wetlands -> 5
        12: 6, 14: 6,                        # croplands -> 6
        13: 7,                               # urban -> 7
        15: 8,                               # snow/ice -> 8
        16: 9,                               # barren -> 9
        17: 10,                              # water -> 10
    }

    def __init__(
        self,
        root: str,
        season: Optional[str] = None,
        *,
        modalities: Iterable[Modality] = (
            Modality.SAR,
            Modality.MULTISPECTRAL,
            Modality.OPTICAL_RGB,
        ),
        size: Optional[int] = None,
        preprocess: bool = False,
        igbp_simplified: bool = True,
        max_patches: Optional[int] = None,
    ) -> None:
        info = DownloadInfo(
            "SEN12MS",
            DATASET_URLS["sen12ms"],
            note=(
                "Expected layout: root/ROIs<id>_<season>/{s1,s2,lc}_<scene>/"
                "ROIs..._<s1|s2|lc>_<scene>_p<patch>.tif (GeoTIFF; needs rasterio)."
            ),
        )
        _require_dir(root, info)
        self.root = root
        self.season = season
        self.modalities = [Modality(m) for m in modalities]
        self.size = size
        self.preprocess = preprocess
        self.igbp_simplified = igbp_simplified
        self.max_patches = max_patches
        self._info = info
        # Index of (s1_path, s2_path, lc_path, key) tuples (lazy file reads).
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[str, str, Optional[str], str]]:
        """Scan the SEN12MS tree and pair S1/S2/LC patches by scene+patch key."""
        index: list[tuple[str, str, Optional[str], str]] = []
        for roi in sorted(os.listdir(self.root)):
            roi_dir = os.path.join(self.root, roi)
            if not os.path.isdir(roi_dir):
                continue
            if self.season is not None and self.season.lower() not in roi.lower():
                continue
            # Collect s1/s2/lc patches keyed by the trailing "<scene>_p<patch>".
            s1: dict[str, str] = {}
            s2: dict[str, str] = {}
            lc: dict[str, str] = {}
            for sub in sorted(os.listdir(roi_dir)):
                sub_dir = os.path.join(roi_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                low = sub.lower()
                target = (
                    s1 if low.startswith("s1")
                    else s2 if low.startswith("s2")
                    else lc if low.startswith("lc")
                    else None
                )
                if target is None:
                    continue
                for fn in sorted(os.listdir(sub_dir)):
                    if not fn.lower().endswith(_GEO_EXTS):
                        continue
                    key = self._patch_key(fn)
                    if key is not None:
                        target[key] = os.path.join(sub_dir, fn)
            # Pair by shared key (require both S1 and S2 present).
            for key in sorted(set(s1) & set(s2)):
                index.append((s1[key], s2[key], lc.get(key), f"{roi}:{key}"))
                if self.max_patches is not None and len(index) >= self.max_patches:
                    return index
        return index

    @staticmethod
    def _patch_key(filename: str) -> Optional[str]:
        """Extract ``<scene>_p<patch>`` from a SEN12MS filename.

        Filenames look like ``ROIs1158_spring_s1_0_p100.tif``; the modality token
        (``s1``/``s2``/``lc``) is stripped so the same physical patch across
        modalities maps to the same key.
        """
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        # Drop the modality token if present so keys align across modalities.
        norm = [p for p in parts if p.lower() not in ("s1", "s2", "lc")]
        if len(norm) < 2:
            return None
        return "_".join(norm[-2:])  # "<scene>_p<patch>"

    def __len__(self) -> int:
        return len(self._index)

    def _read_label(self, lc_path: Optional[str]) -> int:
        """Derive a single IGBP class id from a land-cover patch (majority vote)."""
        if lc_path is None:
            return -1
        lc = load_image_any(lc_path)
        # SEN12MS LC is multi-layer; band 0 is the IGBP class map.
        igbp = lc[0].astype(np.int64)
        vals, counts = np.unique(igbp[igbp > 0], return_counts=True)
        if vals.size == 0:
            return -1
        cls = int(vals[int(np.argmax(counts))])
        if self.igbp_simplified:
            return self.IGBP17_TO_SIMPLIFIED10.get(cls, 0)
        return cls

    def _emit(
        self, s1_path: str, s2_path: str, lc_path: Optional[str], key: str
    ) -> list[Sample]:
        label = self._read_label(lc_path)
        out: list[Sample] = []

        def _post(arr: np.ndarray, modality: Modality) -> np.ndarray:
            if self.preprocess and self.size is not None:
                from .preprocessing import preprocess as _pp

                return _pp(arr, modality, size=self.size)
            if self.size is not None:
                from .preprocessing import resize_image

                return resize_image(arr, self.size)
            return arr.astype(np.float32, copy=False)

        s2 = None
        for m in self.modalities:
            if m == Modality.SAR:
                arr = load_image_any(s1_path)
                out.append(self._mk(key, _post(arr, m), m, label, s1_path))
            elif m == Modality.MULTISPECTRAL:
                s2 = load_image_any(s2_path) if s2 is None else s2
                out.append(self._mk(key, _post(s2, m), m, label, s2_path))
            elif m == Modality.OPTICAL_RGB:
                s2 = load_image_any(s2_path) if s2 is None else s2
                rgb = to_rgb(s2)
                out.append(
                    self._mk(key, _post(rgb, m), m, label, s2_path, derived=True)
                )
        return out

    def _mk(
        self,
        key: str,
        img: np.ndarray,
        modality: Modality,
        label: int,
        path: str,
        *,
        derived: bool = False,
    ) -> Sample:
        return Sample(
            id=f"{modality.value}:{key}",
            image=img,
            modality=modality,
            label=label,
            location_id=key,
            meta={"path": path, "derived": derived, "season": self.season},
        )

    def __iter__(self) -> Iterator[Sample]:
        for s1, s2, lc, key in self._index:
            yield from self._emit(s1, s2, lc, key)

    def to_list(self, limit: Optional[int] = None) -> list[Sample]:
        """Materialise samples (each patch yields up to 3 co-registered samples)."""
        out: list[Sample] = []
        for s1, s2, lc, key in self._index:
            out.extend(self._emit(s1, s2, lc, key))
            if limit is not None and len(out) >= limit:
                return out[:limit]
        return out


# ---------------------------------------------------------------------------
# MultiModalDataset + split builders.
# ---------------------------------------------------------------------------
class MultiModalDataset:
    """Group :class:`Sample` objects by ``location_id`` across modalities.

    Wraps a flat ``list[Sample]`` (typically the union of several adapters) and
    provides the cross-modal bookkeeping the retrieval evaluation needs:

    * ``by_location`` -- ``location_id -> {modality: [samples]}`` groupings, the
      source of true cross-modal positive pairs (same place, different sensors).
    * ``by_modality`` -- ``modality -> [samples]`` for same-modal galleries.
    * split builders for same-modal and cross-modal (query, gallery) evaluation.

    Parameters
    ----------
    samples:
        Flat list of samples. Samples whose ``location_id`` is ``None`` are kept
        but contribute only to same-modal (class-based) evaluation.
    """

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = list(samples)
        self.by_modality: dict[Modality, list[Sample]] = defaultdict(list)
        self.by_location: dict[str, dict[Modality, list[Sample]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for s in self.samples:
            self.by_modality[s.modality].append(s)
            if s.location_id is not None:
                self.by_location[s.location_id][s.modality].append(s)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def modalities(self) -> list[Modality]:
        """Modalities present in the dataset (sorted by value)."""
        return sorted(self.by_modality.keys(), key=lambda m: m.value)

    def cross_modal_pairs(
        self, modality_a: Modality, modality_b: Modality
    ) -> list[tuple[Sample, Sample]]:
        """Return co-located (sample_a, sample_b) pairs across two modalities.

        These are the *ground-truth cross-modal positives* (same ``location_id``)
        used for contrastive training and for geographic-correspondence
        relevance. All A-B combinations at a shared location are returned.
        """
        modality_a, modality_b = Modality(modality_a), Modality(modality_b)
        pairs: list[tuple[Sample, Sample]] = []
        for groups in self.by_location.values():
            a_list = groups.get(modality_a, [])
            b_list = groups.get(modality_b, [])
            for a in a_list:
                for b in b_list:
                    pairs.append((a, b))
        return pairs

    def make_same_modal_split(
        self,
        modality: Modality,
        *,
        query_frac: float = 0.2,
        class_balanced_gallery: bool = True,
        seed: int = 0,
    ) -> tuple[list[Sample], list[Sample]]:
        """Build a (query, gallery) split within a single modality.

        Convenience wrapper over :func:`make_query_gallery_split` restricted to
        ``modality`` -- the same-modal retrieval protocol (e.g. SAR->SAR).
        """
        modality = Modality(modality)
        return make_query_gallery_split(
            self.by_modality.get(modality, []),
            query_frac=query_frac,
            class_balanced_gallery=class_balanced_gallery,
            seed=seed,
        )

    def make_cross_modal_split(
        self,
        query_modality: Modality,
        gallery_modality: Modality,
        *,
        query_frac: float = 0.2,
        class_balanced_gallery: bool = True,
        seed: int = 0,
    ) -> tuple[list[Sample], list[Sample]]:
        """Build a cross-modal (query, gallery) split (e.g. optical -> SAR).

        Queries are drawn from ``query_modality`` and the gallery from
        ``gallery_modality``; the two are disjoint modalities so every retrieval
        is genuinely cross-modal. Relevance is by class (and, where
        ``location_id`` matches, by geographic correspondence). Each modality is
        split independently with the same ``seed`` for reproducibility.
        """
        query_modality = Modality(query_modality)
        gallery_modality = Modality(gallery_modality)
        queries, _ = make_query_gallery_split(
            self.by_modality.get(query_modality, []),
            query_frac=query_frac,
            class_balanced_gallery=False,
            seed=seed,
        )
        _, gallery = make_query_gallery_split(
            self.by_modality.get(gallery_modality, []),
            query_frac=query_frac,
            class_balanced_gallery=class_balanced_gallery,
            seed=seed + 1,
        )
        return queries, gallery

    def pooled_gallery(
        self, *, exclude: Optional[Iterable[Sample]] = None
    ) -> list[Sample]:
        """Return all samples pooled across modalities (mixed cross-modal gallery).

        The mixed gallery (all modalities together) is used to score retrieval
        against a heterogeneous archive, as in the PS-11 evaluation protocol.
        """
        excl_ids = {s.id for s in exclude} if exclude is not None else set()
        return [s for s in self.samples if s.id not in excl_ids]


def make_query_gallery_split(
    samples: list[Sample],
    query_frac: float = 0.2,
    class_balanced_gallery: bool = True,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into (queries, gallery), stratified by class.

    Implements the evaluation protocol from ``research/01_datasets.md``: a
    fraction ``query_frac`` of each class becomes the query set and the remainder
    the gallery, so every query class is represented in the gallery (otherwise
    F1@K would be trivially zero for that class). Deterministic given ``seed``.

    Parameters
    ----------
    samples:
        Samples to split (typically one modality's samples).
    query_frac:
        Fraction of each class assigned to the query set (``0 < frac < 1``).
    class_balanced_gallery:
        If ``True``, truncate the gallery so every class contributes the same
        number of items (the minimum per-class gallery count). This makes F1@K
        comparable across classes and matches the "macro-averaged" scoring.
    seed:
        RNG seed for the per-class shuffle.

    Returns
    -------
    (queries, gallery):
        Two disjoint lists of samples.

    Raises
    ------
    ValueError
        If ``query_frac`` is not in ``(0, 1)``.
    """
    if not 0.0 < query_frac < 1.0:
        raise ValueError(f"query_frac must be in (0, 1), got {query_frac}")
    if not samples:
        return [], []

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    queries: list[Sample] = []
    gallery_by_class: dict[int, list[Sample]] = {}
    for label, items in by_class.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_query = max(1, int(round(len(items) * query_frac)))
        # Guarantee at least one gallery item per class when possible.
        if len(items) > 1:
            n_query = min(n_query, len(items) - 1)
        q_idx = idx[:n_query]
        g_idx = idx[n_query:]
        queries.extend(items[i] for i in q_idx)
        gallery_by_class[label] = [items[i] for i in g_idx]

    if class_balanced_gallery and gallery_by_class:
        non_empty = [v for v in gallery_by_class.values() if v]
        if non_empty:
            min_count = min(len(v) for v in non_empty)
            gallery: list[Sample] = []
            for items in gallery_by_class.values():
                gallery.extend(items[:min_count])
        else:
            gallery = []
    else:
        gallery = [s for items in gallery_by_class.values() for s in items]

    return queries, gallery


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
def get_dataset(name: str, **kwargs) -> object:
    """Instantiate a dataset adapter by name.

    A tiny registry so callers / config files can refer to datasets by string.

    Supported names (case-insensitive):
    ``"folder"``, ``"eurosat"``, ``"sen12ms"``, ``"synthetic"``.

    Parameters
    ----------
    name:
        Dataset key.
    **kwargs:
        Forwarded to the adapter constructor (or to
        :func:`~xsretrieval.data.synthetic.make_synthetic_multimodal` for
        ``"synthetic"``, in which case a :class:`MultiModalDataset` is returned).

    Returns
    -------
    object
        The constructed adapter (or :class:`MultiModalDataset` for synthetic).

    Raises
    ------
    KeyError
        If ``name`` is not a known dataset.
    """
    key = name.lower()
    if key == "folder":
        return FolderDataset(**kwargs)
    if key == "eurosat":
        return EuroSATDataset(**kwargs)
    if key == "sen12ms":
        return SEN12MSDataset(**kwargs)
    if key == "synthetic":
        from .synthetic import make_synthetic_multimodal

        samples = make_synthetic_multimodal(**kwargs)
        return MultiModalDataset(samples)
    raise KeyError(
        f"unknown dataset {name!r}; available: "
        "folder, eurosat, sen12ms, synthetic"
    )
