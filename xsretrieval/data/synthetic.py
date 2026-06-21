"""Synthetic multi-modal remote-sensing data with recoverable cross-modal structure.

The smoke test for the whole ``xsretrieval`` pipeline needs data that is
*solvable* -- a simple feature extractor must achieve clearly-above-chance F1 on
**both** same-modal and cross-modal retrieval -- yet realistic enough to exercise
modality-specific code paths (SAR speckle, MS band structure, RGB colour).

Design (the important part)
---------------------------
Each (class, location) pair owns a **shared low-dimensional latent vector**. From
that single latent we render a DISTINCT image per modality, but in a way that the
*same* underlying structure is recoverable:

* A small set of smooth 2-D spatial basis fields (low-frequency cosine bumps) is
  combined with **class-determined low-frequency coefficients** to produce a
  shared "scene layout". The class controls the coarse texture/structure, so a
  blurred-and-downsampled descriptor (what a weak feature extractor sees)
  separates classes.
* Each modality maps that shared layout through a **modality-specific but fixed**
  rendering function:
    - OPTICAL_RGB -- the layout drives 3 colour channels via a per-class colour
      palette (class also controls hue), values in ``[0, 1]``.
    - MULTISPECTRAL -- the layout is projected onto 13 bands via a fixed spectral
      mixing matrix plus class-dependent spectral signature; band noise added.
    - SAR -- the layout modulates VV/VH backscatter in dB, then realistic
      multiplicative Gamma speckle is applied (so SAR is genuinely noisy).
* Because every modality of a given location is generated from the **same
  latent**, the layout correlates across modalities -> cross-modal positives are
  real. Because the latent is **class-structured**, same-class items (even at
  different locations / modalities) share coarse structure -> class retrieval
  works.

A per-location random offset adds intra-class variation so retrieval is
non-trivial (not a single template per class). Everything is deterministic given
``seed`` and uses only numpy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .modalities import MODALITY_CHANNELS, Modality, Sample

__all__ = [
    "make_synthetic_multimodal",
    "make_synthetic_embeddings",
]


# ---------------------------------------------------------------------------
# Shared latent / layout generation.
# ---------------------------------------------------------------------------
def _spatial_basis(size: int, n_basis: int) -> np.ndarray:
    """Return ``n_basis`` smooth low-frequency 2-D fields, shape ``(n_basis, H, W)``.

    Each field is a separable product of low-order cosines. Low frequency is
    deliberate: a weak feature extractor (downsample + flatten) can only see
    coarse structure, so the recoverable signal must live at low spatial
    frequencies. Fields are normalised to roughly zero mean, unit scale.
    """
    ys = np.linspace(0.0, np.pi, size, dtype=np.float32)
    xs = np.linspace(0.0, np.pi, size, dtype=np.float32)
    fields = np.empty((n_basis, size, size), dtype=np.float32)
    for b in range(n_basis):
        # Frequencies kept small (1..3) so structure is coarse/recoverable.
        fy = 1 + (b % 3)
        fx = 1 + ((b // 3) % 3)
        phase = 0.5 * b
        field = np.outer(np.cos(fy * ys + phase), np.cos(fx * xs + phase))
        fields[b] = field.astype(np.float32)
    # Standardise each field.
    fields -= fields.mean(axis=(1, 2), keepdims=True)
    norm = fields.std(axis=(1, 2), keepdims=True) + 1e-6
    fields /= norm
    return fields


def _class_layout_coeffs(
    n_classes: int, n_basis: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-class coefficients over the spatial basis, shape ``(n_classes, n_basis)``.

    Each class gets a well-separated random coefficient vector (the coarse
    "shape signature" of the class). Separation is encouraged by sampling from a
    standard normal and L2-normalising so classes spread over the hypersphere.
    """
    coeffs = rng.normal(size=(n_classes, n_basis)).astype(np.float32)
    coeffs /= np.linalg.norm(coeffs, axis=1, keepdims=True) + 1e-6
    return coeffs


def _location_latent(
    class_coeffs: np.ndarray,
    rng: np.random.Generator,
    *,
    jitter: float = 0.25,
) -> np.ndarray:
    """Per-location latent = class signature + small random offset.

    The offset (scaled by ``jitter``) creates intra-class variation so retrieval
    is non-trivial, while the class signature keeps same-class locations close.
    """
    offset = rng.normal(scale=jitter, size=class_coeffs.shape).astype(np.float32)
    return class_coeffs + offset


def _render_layout(latent: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Combine a latent coefficient vector with the spatial basis -> ``(H, W)``.

    This is the shared scene layout that every modality is rendered from.
    """
    layout = np.tensordot(latent, basis, axes=(0, 0))  # (H, W)
    # Squash to [0, 1] with a smooth sigmoid so all modalities start from a
    # bounded "albedo-like" field.
    return (1.0 / (1.0 + np.exp(-layout))).astype(np.float32)


# ---------------------------------------------------------------------------
# Modality-specific renderers (deterministic mapping + modality noise).
# ---------------------------------------------------------------------------
def _render_rgb(
    layout: np.ndarray,
    class_id: int,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render optical RGB from the shared layout + a per-class colour palette.

    Class controls the hue (a point on a colour wheel) so colour is a strong
    class cue; the layout modulates intensity so structure is shared with the
    other modalities. Output in ``[0, 1]``, shape ``(3, H, W)``.
    """
    # Per-class base colour on a smooth wheel (deterministic in class_id).
    hue = 2.0 * np.pi * (class_id / max(1, n_classes))
    palette = np.array(
        [
            0.5 + 0.5 * np.cos(hue),
            0.5 + 0.5 * np.cos(hue + 2.094),  # +120 deg
            0.5 + 0.5 * np.cos(hue + 4.188),  # +240 deg
        ],
        dtype=np.float32,
    )
    # Modulate the palette by the layout (structure) and add mild noise.
    rgb = palette[:, None, None] * (0.4 + 0.6 * layout[None, :, :])
    rgb += rng.normal(scale=0.03, size=rgb.shape).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _render_ms(
    layout: np.ndarray,
    class_id: int,
    n_classes: int,
    rng: np.random.Generator,
    mix: np.ndarray,
) -> np.ndarray:
    """Render a 13-band MS image via a fixed spectral mix + class signature.

    The shared layout is lifted into 13 bands by a *fixed* mixing matrix ``mix``
    (shared across all samples, so the cross-modal correlation is stable), then a
    class-dependent spectral signature scales each band (the spectral fingerprint
    of the land-cover class). Per-band Gaussian noise is added. Output ``(13, H, W)``
    in a reflectance-like ``[0, 1]`` range.
    """
    n_bands = MODALITY_CHANNELS[Modality.MULTISPECTRAL]
    # Class spectral signature: smooth per-band gains (deterministic in class).
    band_idx = np.arange(n_bands, dtype=np.float32)
    sig = 0.5 + 0.5 * np.sin(
        0.7 * band_idx + 2.0 * np.pi * class_id / max(1, n_classes)
    )
    # mix: (n_bands, 2), NON-NEGATIVE so every band is monotonically increasing
    # in the shared layout. This guarantees the layout polarity is preserved
    # across modalities (RGB/SAR/MS all brighten with structure), which is what
    # makes cross-modal retrieval solvable by a simple structural descriptor.
    # Build a 2-vector field [layout, layout^1.5] so bands are correlated but
    # not identical, then apply the class spectral signature for separability.
    feat = np.stack([layout, layout**1.5], axis=0)  # (2, H, W)
    ms = np.tensordot(np.abs(mix), feat, axes=(1, 0))  # (n_bands, H, W) >= 0
    ms = ms * sig[:, None, None]
    ms += rng.normal(scale=0.02, size=ms.shape).astype(np.float32)
    # Normalise into a reflectance-like positive range (sign-preserving:
    # subtracting the global min keeps the layout's high/low structure intact).
    ms = (ms - ms.min()) / (np.ptp(ms) + 1e-6)
    return ms.astype(np.float32)


def _render_sar(
    layout: np.ndarray,
    class_id: int,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render SAR VV/VH in dB from the shared layout, with Gamma speckle.

    The layout modulates the mean backscatter (a class-dependent offset shifts
    the dB level), then **multiplicative Gamma speckle** is applied in the linear
    domain before converting back to dB -- giving genuinely speckled SAR that the
    SAR preprocessing/augmentation paths must cope with. Output ``(2, H, W)`` in dB.
    """
    # Class shifts the mean dB level (separates classes), layout adds structure.
    base_vv = -12.0 + 6.0 * (layout - 0.5) + 3.0 * np.cos(
        2.0 * np.pi * class_id / max(1, n_classes)
    )
    base_vh = base_vv - 6.0 - 1.5 * layout  # VH typically lower than VV
    db = np.stack([base_vv, base_vh], axis=0).astype(np.float32)

    # Apply multiplicative Gamma speckle in linear power, then back to dB.
    linear = np.power(10.0, db / 10.0)
    looks = 4  # moderate speckle level
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=linear.shape)
    linear = linear * np.maximum(speckle, 1e-6)
    db_speckled = 10.0 * np.log10(np.maximum(linear, 1e-8))
    return db_speckled.astype(np.float32)


_RENDERERS = {
    Modality.OPTICAL_RGB: "rgb",
    Modality.MULTISPECTRAL: "ms",
    Modality.SAR: "sar",
}


def make_synthetic_multimodal(
    n_classes: int = 10,
    per_class_per_modality: int = 20,
    modalities: Sequence[Modality] = (
        Modality.OPTICAL_RGB,
        Modality.MULTISPECTRAL,
        Modality.SAR,
    ),
    size: int = 64,
    seed: int = 0,
    *,
    n_basis: int = 9,
    jitter: float = 0.25,
) -> list[Sample]:
    """Generate a solvable synthetic multi-modal retrieval dataset.

    Produces ``n_classes * per_class_per_modality * len(modalities)`` samples.
    For every (class, location) a shared latent is drawn and each requested
    modality is rendered from it, so:

    * **Same-modal** retrieval works because same-class items share coarse
      structure and (for RGB) colour.
    * **Cross-modal** retrieval works because all modalities of a location are
      rendered from the *same* latent, and same-class locations share the class
      signature -- the modalities live in correlated, class-structured spaces.

    Each location yields one sample per modality, and they share a common
    ``location_id`` (``"cls{c}_loc{l}"``), giving exact cross-modal positives in
    addition to class-level relevance.

    Parameters
    ----------
    n_classes:
        Number of semantic classes.
    per_class_per_modality:
        Number of locations per class (each location -> one sample per modality).
    modalities:
        Which modalities to render. Supported: OPTICAL_RGB, MULTISPECTRAL, SAR.
        (DEM / hyperspectral are rendered with the MS path as a fallback.)
    size:
        Spatial size of each image (square).
    seed:
        Master RNG seed (full determinism).
    n_basis:
        Number of spatial basis fields (controls structural richness).
    jitter:
        Intra-class latent jitter (higher -> harder retrieval).

    Returns
    -------
    list[Sample]
        Flat list of samples (mixed modalities), each with ``label`` and
        ``location_id`` set, and ``image`` of shape matching the modality's
        canonical channel count.
    """
    modalities = [Modality(m) for m in modalities]
    master = np.random.default_rng(seed)

    # Fixed, dataset-wide structures (shared across all samples).
    basis = _spatial_basis(size, n_basis)
    class_coeffs = _class_layout_coeffs(n_classes, n_basis, master)
    # Fixed spectral mixing matrix for MS (13 bands from a 2-D layout feature).
    ms_mix = master.normal(
        size=(MODALITY_CHANNELS[Modality.MULTISPECTRAL], 2)
    ).astype(np.float32)

    samples: list[Sample] = []
    for c in range(n_classes):
        for loc in range(per_class_per_modality):
            location_id = f"cls{c}_loc{loc}"
            # Per-location latent shared by every modality (derive a stable
            # per-location RNG so results are reproducible regardless of order).
            loc_rng = np.random.default_rng(
                (seed * 1_000_003 + c * 7919 + loc) & 0xFFFFFFFF
            )
            latent = _location_latent(class_coeffs[c], loc_rng, jitter=jitter)
            layout = _render_layout(latent, basis)

            for m in modalities:
                # Independent noise stream per (location, modality).
                mod_rng = np.random.default_rng(
                    (
                        seed * 2_654_435_761
                        + c * 40_503
                        + loc * 257
                        + hash(m.value) % 100_003
                    )
                    & 0xFFFFFFFF
                )
                kind = _RENDERERS.get(m, "ms")
                if kind == "rgb":
                    img = _render_rgb(layout, c, n_classes, mod_rng)
                elif kind == "sar":
                    img = _render_sar(layout, c, n_classes, mod_rng)
                else:  # ms (and DEM/HS fallback)
                    img = _render_ms(layout, c, n_classes, mod_rng, ms_mix)
                    if MODALITY_CHANNELS.get(m, 0) == 1:
                        img = img[:1]  # DEM fallback -> single band

                sid = f"{m.value}_{location_id}"
                samples.append(
                    Sample(
                        id=sid,
                        image=img,
                        modality=m,
                        label=c,
                        location_id=location_id,
                        meta={"synthetic": True},
                    )
                )
    return samples


def make_synthetic_embeddings(
    n_classes: int = 10,
    per_class_per_modality: int = 20,
    modalities: Sequence[Modality] = (
        Modality.OPTICAL_RGB,
        Modality.MULTISPECTRAL,
        Modality.SAR,
    ),
    dim: int = 256,
    seed: int = 0,
    *,
    class_sep: float = 4.0,
    modality_shift: float = 0.5,
    noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate class-structured, L2-normalized embeddings for fast unit tests.

    Skips image encoding entirely: returns ready ``(N, D)`` embeddings whose
    class structure makes both same-modal and cross-modal retrieval solvable,
    plus matching labels and modality codes. Each class has a random prototype on
    the hypersphere; each modality adds a small fixed shift (the "modality gap")
    so cross-modal retrieval is harder than same-modal but still works -- exactly
    the regime the real system targets.

    Parameters
    ----------
    n_classes:
        Number of classes.
    per_class_per_modality:
        Samples per class per modality.
    modalities:
        Modalities to emit (their order defines the integer codes returned).
    dim:
        Embedding dimensionality.
    seed:
        RNG seed.
    class_sep:
        Scale of the class-prototype separation (higher -> easier).
    modality_shift:
        Magnitude of the per-modality offset (higher -> larger modality gap,
        harder cross-modal).
    noise:
        Std of per-sample Gaussian noise around the prototype.

    Returns
    -------
    (embeddings, labels, modality_codes):
        * ``embeddings``: ``(N, D)`` float32, L2-normalized.
        * ``labels``: ``(N,)`` int64 class ids.
        * ``modality_codes``: ``(N,)`` int64, index into ``modalities``.
    """
    modalities = [Modality(m) for m in modalities]
    rng = np.random.default_rng(seed)

    class_proto = rng.normal(size=(n_classes, dim)).astype(np.float32)
    class_proto /= np.linalg.norm(class_proto, axis=1, keepdims=True) + 1e-6
    class_proto *= class_sep
    modality_offset = rng.normal(
        size=(len(modalities), dim)
    ).astype(np.float32) * modality_shift

    embs: list[np.ndarray] = []
    labels: list[int] = []
    mod_codes: list[int] = []
    for c in range(n_classes):
        for mi, _m in enumerate(modalities):
            center = class_proto[c] + modality_offset[mi]
            block = center[None, :] + rng.normal(
                scale=noise, size=(per_class_per_modality, dim)
            ).astype(np.float32)
            embs.append(block)
            labels.extend([c] * per_class_per_modality)
            mod_codes.extend([mi] * per_class_per_modality)

    embeddings = np.concatenate(embs, axis=0).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6
    return (
        embeddings,
        np.asarray(labels, dtype=np.int64),
        np.asarray(mod_codes, dtype=np.int64),
    )
