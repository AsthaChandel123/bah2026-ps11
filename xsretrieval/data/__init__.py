"""Data layer for the ``xsretrieval`` cross-modal satellite retrieval package.

This subpackage owns everything between raw sensor files and ready-to-encode
numpy arrays:

* **Modalities & containers** -- :class:`Modality`, :class:`Sample`,
  :data:`MODALITY_CHANNELS`, normalization constants and band helpers.
* **Preprocessing** -- :func:`preprocess` plus SAR despeckling and previews.
* **Augmentation** -- per-modality, co-registration-preserving augmentations
  (:func:`build_augmentation`, :class:`PairedAugmentor`, ...).
* **Datasets** -- lazy, gracefully-degrading adapters (:class:`FolderDataset`,
  :class:`EuroSATDataset`, :class:`SEN12MSDataset`, :class:`MultiModalDataset`)
  and the query/gallery split builder.
* **Synthetic data** -- :func:`make_synthetic_multimodal` /
  :func:`make_synthetic_embeddings` for smoke tests and unit tests.
* **Sampling** -- :class:`PKModalitySampler` (modality-balanced P x K batches).

All public names are re-exported here so callers can simply do
``from xsretrieval.data import Sample, preprocess, make_synthetic_multimodal``.

No heavy dependency (torch / faiss / transformers / timm / rasterio) is imported
at module load time; only numpy (a hard dependency) is required to import this
package.
"""

from __future__ import annotations

from .modalities import (
    MODALITY_CHANNELS,
    NORMALIZATION_STATS,
    RGB_MEAN,
    RGB_STD,
    S2_BANDS,
    S2_RGB_BAND_INDICES,
    SAR_DB_CLIP,
    SAR_MEAN,
    SAR_STD,
    MS_MEAN,
    MS_STD,
    DEM_MEAN,
    DEM_STD,
    Modality,
    Sample,
    get_normalization,
    to_rgb,
)
from .preprocessing import (
    despeckle_lee,
    ensure_channels,
    preprocess,
    resize_image,
    sar_to_db,
    to_uint8_preview,
)
from .augmentations import (
    Augmentation,
    Compose,
    PairedAugmentor,
    build_augmentation,
    cross_modal_mixup,
    modality_dropout,
    # SAR
    sar_additive_speckle,
    sar_db_jitter,
    sar_gamma_speckle,
    sar_pol_dropout,
    sar_random_despeckle,
    # optical
    optical_color_jitter,
    optical_gaussian_blur,
    optical_random_erasing,
    optical_random_grayscale,
    # multispectral
    ms_band_dropout,
    ms_band_gain_bias,
    ms_ndvi_jitter,
    ms_spectral_tube_mask,
    # geometric
    apply_geometric,
    geom_hflip,
    geom_random_resized_crop,
    geom_rot90,
    geom_small_rotation,
    geom_vflip,
)
from .datasets import (
    DATASET_URLS,
    EuroSATDataset,
    FolderDataset,
    MultiModalDataset,
    SEN12MSDataset,
    get_dataset,
    load_image_any,
    make_query_gallery_split,
)
from .synthetic import make_synthetic_embeddings, make_synthetic_multimodal
from .samplers import PKModalitySampler

__all__ = [
    # modalities & containers
    "Modality",
    "Sample",
    "MODALITY_CHANNELS",
    "NORMALIZATION_STATS",
    "RGB_MEAN",
    "RGB_STD",
    "MS_MEAN",
    "MS_STD",
    "SAR_MEAN",
    "SAR_STD",
    "SAR_DB_CLIP",
    "DEM_MEAN",
    "DEM_STD",
    "S2_BANDS",
    "S2_RGB_BAND_INDICES",
    "get_normalization",
    "to_rgb",
    # preprocessing
    "preprocess",
    "resize_image",
    "ensure_channels",
    "sar_to_db",
    "despeckle_lee",
    "to_uint8_preview",
    # augmentation
    "Augmentation",
    "Compose",
    "build_augmentation",
    "PairedAugmentor",
    "modality_dropout",
    "cross_modal_mixup",
    "apply_geometric",
    "geom_hflip",
    "geom_vflip",
    "geom_rot90",
    "geom_small_rotation",
    "geom_random_resized_crop",
    "sar_gamma_speckle",
    "sar_additive_speckle",
    "sar_db_jitter",
    "sar_pol_dropout",
    "sar_random_despeckle",
    "optical_color_jitter",
    "optical_random_grayscale",
    "optical_gaussian_blur",
    "optical_random_erasing",
    "ms_band_dropout",
    "ms_spectral_tube_mask",
    "ms_band_gain_bias",
    "ms_ndvi_jitter",
    # datasets
    "FolderDataset",
    "EuroSATDataset",
    "SEN12MSDataset",
    "MultiModalDataset",
    "make_query_gallery_split",
    "get_dataset",
    "load_image_any",
    "DATASET_URLS",
    # synthetic
    "make_synthetic_multimodal",
    "make_synthetic_embeddings",
    # samplers
    "PKModalitySampler",
]
