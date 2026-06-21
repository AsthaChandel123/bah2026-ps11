"""Abstract backbone interface, registry, and shared helpers.

This module is the contract every feature-extraction backbone in
``xsretrieval`` implements. It is intentionally dependency-light: only
``numpy`` is imported at module scope so that ``import
xsretrieval.models.backbones.base`` succeeds on a bare-numpy install.
Heavy dependencies (``torch``, ``transformers``, ``timm``, ``open_clip``,
``huggingface_hub``) are imported lazily *inside* methods of the concrete
backbones, never here.

Key concepts
------------
* :class:`Backbone` -- the abstract base. A backbone maps a batch of images
  (for a given :class:`~xsretrieval.data.modalities.Modality`) to an
  ``(B, embed_dim)`` ``float32`` array of **L2-normalized** embeddings.
* ``REGISTRY`` / :func:`register_backbone` / :func:`get_backbone` -- a string
  registry so the pipeline / config can construct a backbone by name. If the
  requested backbone cannot be built (e.g. its optional dependency is missing
  or a weight download fails), :func:`get_backbone` logs a warning and falls
  back to the always-works :class:`FallbackBackbone` so the pipeline never
  hard-crashes.

Embedding convention (shared across all teams)
----------------------------------------------
``Backbone.embed(images, modality) -> np.ndarray`` of shape
``(B, embed_dim)``, dtype ``float32``, **L2-normalized** along ``axis=1``.
Boundaries between teams are plain ``numpy`` arrays.
"""

from __future__ import annotations

import abc
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import torch

    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = [
    "Backbone",
    "REGISTRY",
    "register_backbone",
    "get_backbone",
    "l2_normalize",
]


# ---------------------------------------------------------------------------
# Modality helpers
# ---------------------------------------------------------------------------
# We import ``Modality`` / ``MODALITY_CHANNELS`` lazily from the data team's
# module. Final code imports them normally, but importing inside a helper keeps
# this module importable for isolated unit tests *before* the data package
# lands, and avoids a hard import cycle (data <-> models) during integration.


def _modality_name(modality: "Modality | str | None") -> str:
    """Return a lowercase string identifier for *modality*.

    Accepts a :class:`~xsretrieval.data.modalities.Modality` enum member, a raw
    string, or ``None`` (treated as ``"unknown"``). This is deliberately
    permissive so backbones can branch on modality without taking a hard
    dependency on the exact enum member spellings.
    """
    if modality is None:
        return "unknown"
    # Enum members expose ``.name`` (and usually ``.value``); prefer ``.value``
    # when it is a string, else fall back to ``.name``, else ``str()``.
    value = getattr(modality, "value", None)
    if isinstance(value, str):
        return value.lower()
    name = getattr(modality, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(modality).lower()


def _is_sar(modality: "Modality | str | None") -> bool:
    return "sar" in _modality_name(modality)


def _is_multispectral(modality: "Modality | str | None") -> bool:
    name = _modality_name(modality)
    return ("multispectral" in name) or name in {"ms", "msi", "s2", "sentinel2"}


def _is_rgb(modality: "Modality | str | None") -> bool:
    name = _modality_name(modality)
    return ("rgb" in name) or ("optical" in name)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize *x* along *axis*; returns a contiguous ``float32`` array.

    Zero (or near-zero) vectors are left at (near) zero rather than producing
    NaNs, thanks to the ``eps`` floor on the norm.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return np.ascontiguousarray(x / norm, dtype=np.float32)


# ---------------------------------------------------------------------------
# Abstract backbone
# ---------------------------------------------------------------------------
class Backbone(abc.ABC):
    """Abstract feature-extraction backbone.

    Subclasses must set :attr:`name`, :attr:`embed_dim`,
    :attr:`supported_modalities`, and implement :meth:`_embed_batch`. The public
    :meth:`embed` wraps :meth:`_embed_batch` with input normalization (numpy /
    torch / list-of-samples coercion) and a final L2-normalization, so concrete
    backbones only have to produce raw ``(B, embed_dim)`` features.

    Attributes
    ----------
    name:
        Human-readable / registry name of the backbone.
    embed_dim:
        Output embedding dimensionality (width of the returned vectors).
    supported_modalities:
        Set of :class:`~xsretrieval.data.modalities.Modality` this backbone can
        natively or via adaptation embed. ``None`` entries are not allowed; an
        empty set is interpreted as "all modalities supported".
    device:
        Torch device string (``"cpu"`` by default). The package must run
        CPU-only; backbones honor this when a torch model is involved.
    """

    name: str = "backbone"
    embed_dim: int = 0
    supported_modalities: set["Modality"] = set()
    device: str = "cpu"

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Backbone":
        """Construct a backbone from a config ``dict``.

        Recognized keys are forwarded as keyword arguments to ``__init__``;
        ``"name"`` and ``"type"`` are ignored here (they select *which* class to
        build and are consumed by :func:`get_backbone`). Unknown keys are passed
        through so subclasses can accept extra options.
        """
        kwargs = {k: v for k, v in cfg.items() if k not in {"name", "type"}}
        return cls(**kwargs)  # type: ignore[call-arg]

    # -- public API ---------------------------------------------------------
    def embed(
        self,
        images: Any,
        modality: "Modality | str | None" = None,
    ) -> np.ndarray:
        """Embed a batch of images for *modality*.

        Parameters
        ----------
        images:
            One of:

            * a ``numpy`` array of shape ``(B, C, H, W)`` or ``(C, H, W)``;
            * a ``torch.Tensor`` of the same layout;
            * a list/sequence of per-sample images (each ``(C, H, W)``), or a
              list of ``Sample``-like objects exposing an ``.image`` attribute.
        modality:
            The modality the images belong to. Used for channel adaptation
            (e.g. SAR -> pseudo-RGB). ``None`` is tolerated (no adaptation
            assumptions beyond channel count).

        Returns
        -------
        numpy.ndarray
            ``(B, embed_dim)`` ``float32`` array, L2-normalized along axis 1.
        """
        batch = self._coerce_to_numpy_batch(images)  # (B, C, H, W) float32
        if batch.shape[0] == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        feats = self._embed_batch(batch, modality)
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(
                f"{self.name}._embed_batch must return a 2-D array, "
                f"got shape {feats.shape!r}"
            )
        return l2_normalize(feats, axis=1)

    # -- abstract -----------------------------------------------------------
    @abc.abstractmethod
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        """Compute raw (un-normalized) ``(B, embed_dim)`` features.

        *batch* is guaranteed to be a contiguous ``float32`` ``(B, C, H, W)``
        array with ``B >= 1``. Implementations need not L2-normalize; the public
        :meth:`embed` does that. Channel adaptation per modality is the
        implementation's responsibility (helpers below assist).
        """
        raise NotImplementedError

    # -- input coercion -----------------------------------------------------
    @staticmethod
    def _coerce_to_numpy_batch(images: Any) -> np.ndarray:
        """Coerce supported inputs to a ``(B, C, H, W)`` ``float32`` array."""
        # torch.Tensor (duck-typed to avoid importing torch here).
        if hasattr(images, "detach") and hasattr(images, "cpu") and hasattr(
            images, "numpy"
        ):
            arr = images.detach().cpu().numpy()
            return Backbone._as_bchw(np.asarray(arr, dtype=np.float32))

        if isinstance(images, np.ndarray):
            return Backbone._as_bchw(np.asarray(images, dtype=np.float32))

        # A sequence: list of samples / arrays / tensors.
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            mats: list[np.ndarray] = []
            for item in images:
                mats.append(Backbone._sample_to_chw(item))
            if not mats:
                # Unknown channel count for an empty batch; use 0 channels.
                return np.zeros((0, 0, 0, 0), dtype=np.float32)
            mats = Backbone._pad_stack(mats)
            return np.ascontiguousarray(np.stack(mats, axis=0), dtype=np.float32)

        raise TypeError(
            "images must be a numpy array, torch tensor, or a sequence of "
            f"per-sample images/Samples; got {type(images)!r}"
        )

    @staticmethod
    def _sample_to_chw(item: Any) -> np.ndarray:
        """Extract a ``(C, H, W)`` ``float32`` array from a single item.

        Accepts a raw array/tensor, or a ``Sample``-like object exposing an
        ``.image`` (preferred) or ``.array`` attribute.
        """
        obj = item
        if not isinstance(obj, np.ndarray) and not hasattr(obj, "numpy"):
            # Sample-like: pull the pixel array off a known attribute.
            for attr in ("image", "array", "data", "pixels"):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                    break
        if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
            obj = obj.detach().cpu().numpy()
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim == 2:  # (H, W) -> (1, H, W)
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            # Heuristic: treat (H, W, C) with small trailing dim as channels-last.
            if arr.shape[0] > arr.shape[2] and arr.shape[2] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(
                f"each sample image must be 2-D or 3-D, got shape {arr.shape!r}"
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _as_bchw(arr: np.ndarray) -> np.ndarray:
        """Normalize a single array to ``(B, C, H, W)``."""
        if arr.ndim == 3:  # (C, H, W) -> (1, C, H, W)
            arr = arr[None, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f"image array must be 3-D (C,H,W) or 4-D (B,C,H,W); "
                f"got shape {arr.shape!r}"
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _pad_stack(mats: list[np.ndarray]) -> list[np.ndarray]:
        """Pad a list of ``(C, H, W)`` arrays to a common shape (zero-pad).

        Channel counts are expected to match within a batch; differing spatial
        sizes are zero-padded to the per-batch maximum so a ragged list can be
        stacked. (In practice the data loader resizes uniformly; this is a
        safety net.)
        """
        max_c = max(m.shape[0] for m in mats)
        max_h = max(m.shape[1] for m in mats)
        max_w = max(m.shape[2] for m in mats)
        out: list[np.ndarray] = []
        for m in mats:
            if m.shape == (max_c, max_h, max_w):
                out.append(m)
                continue
            padded = np.zeros((max_c, max_h, max_w), dtype=np.float32)
            padded[: m.shape[0], : m.shape[1], : m.shape[2]] = m
            out.append(padded)
        return out

    # -- channel / modality adaptation helpers ------------------------------
    @staticmethod
    def to_pseudo_rgb(
        batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        """Reduce an arbitrary-channel batch to a 3-channel pseudo-RGB batch.

        This is the shared adaptation used by RGB-only backbones (CLIP family,
        DINOv2, timm) so they can ingest SAR / multispectral inputs:

        * **SAR** (>=2 bands, treated as VV, VH): stack
          ``[VV, VH, VV/VH-ratio]``. The ratio is a classic SAR feature that
          conveys surface/structure information and gives the third channel real
          signal rather than a duplicate.
        * **Multispectral** (Sentinel-2 band order assumed B1..B12/13): pick
          ``[B4 (red), B3 (green), B2 (blue)]`` -> array indices ``[3, 2, 1]``.
          Falls back to the first three channels if there are fewer than four.
        * **RGB / optical** (3 bands): passed through unchanged.
        * **Other channel counts**: if ``C == 1`` the single channel is
          replicated; if ``C >= 3`` and modality is unknown, the first three
          channels are used; otherwise channels are tiled/truncated to 3.

        Parameters
        ----------
        batch:
            ``(B, C, H, W)`` ``float32`` array.
        modality:
            The source modality (drives the band selection).

        Returns
        -------
        numpy.ndarray
            ``(B, 3, H, W)`` ``float32`` array.
        """
        batch = np.asarray(batch, dtype=np.float32)
        b, c, h, w = batch.shape

        if c == 3:
            return np.ascontiguousarray(batch, dtype=np.float32)

        if _is_sar(modality) or (c == 2 and not _is_multispectral(modality)):
            vv = batch[:, 0:1]
            vh = batch[:, 1:2] if c >= 2 else batch[:, 0:1]
            ratio = vv / (np.abs(vh) + 1e-6)
            out = np.concatenate([vv, vh, ratio], axis=1)
            return np.ascontiguousarray(out, dtype=np.float32)

        if _is_multispectral(modality) or c >= 4:
            if c >= 4:
                # Sentinel-2 RGB = B4,B3,B2 -> 0-based indices 3,2,1.
                idx = [3, 2, 1]
            else:
                idx = [min(i, c - 1) for i in (2, 1, 0)]
            out = batch[:, idx, :, :]
            return np.ascontiguousarray(out, dtype=np.float32)

        if c == 1:
            out = np.repeat(batch, 3, axis=1)
            return np.ascontiguousarray(out, dtype=np.float32)

        # c == 2 and multispectral-ish, or any other small count: tile to 3.
        reps = int(np.ceil(3 / c))
        out = np.tile(batch, (1, reps, 1, 1))[:, :3, :, :]
        return np.ascontiguousarray(out, dtype=np.float32)

    def _to_tensor(self, images: Any) -> "torch.Tensor":
        """Coerce *images* to a ``float32`` ``torch.Tensor`` on :attr:`device`.

        Lazy-imports torch. Mirrors :meth:`_coerce_to_numpy_batch` for the
        layout (``(B, C, H, W)``). Used by the torch-backed backbones.
        """
        import torch  # local, lazy

        if isinstance(images, torch.Tensor):
            t = images.to(dtype=torch.float32)
        else:
            batch = self._coerce_to_numpy_batch(images)
            t = torch.from_numpy(batch).to(dtype=torch.float32)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t.to(self.device)

    # -- niceties -----------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mods = (
            "all"
            if not self.supported_modalities
            else sorted(_modality_name(m) for m in self.supported_modalities)
        )
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"embed_dim={self.embed_dim}, device={self.device!r}, "
            f"modalities={mods})"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type[Backbone]] = {}
"""Name -> :class:`Backbone` subclass. Populated by :func:`register_backbone`."""

# Aliases mapping friendly / config names to canonical registry keys. Populated
# lazily by the concrete modules via :func:`register_backbone(name, aliases=...)`.
_ALIASES: dict[str, str] = {}


def register_backbone(
    name: str, aliases: Sequence[str] | None = None
) -> Callable[[type[Backbone]], type[Backbone]]:
    """Class decorator registering a :class:`Backbone` subclass under *name*.

    Parameters
    ----------
    name:
        Canonical registry key (case-insensitive).
    aliases:
        Optional extra names that resolve to the same class.

    Examples
    --------
    >>> @register_backbone("fallback", aliases=["hashfeat"])
    ... class FallbackBackbone(Backbone): ...
    """

    def _decorator(klass: type[Backbone]) -> type[Backbone]:
        key = name.lower()
        REGISTRY[key] = klass
        for alias in aliases or ():
            _ALIASES[alias.lower()] = key
        return klass

    return _decorator


def _resolve_name(name: str) -> str:
    key = name.lower()
    return _ALIASES.get(key, key)


def _import_concrete_backbones() -> None:
    """Import the concrete backbone modules so they self-register.

    Imported lazily (inside :func:`get_backbone`) to keep ``import
    xsretrieval.models.backbones.base`` cheap and side-effect-free. Each module
    only does light top-level work (numpy + registration); heavy deps stay lazy
    inside methods, so these imports are safe on a bare-numpy install.
    """
    from . import fallback  # noqa: F401  (registers FallbackBackbone)

    # The remaining modules are imported defensively: a syntax-clean module
    # registers its class even though its heavy deps are absent. If any fails to
    # import for an unforeseen reason, we keep going -- the fallback is enough to
    # keep the pipeline alive.
    for mod in (
        "precomputed",
        "timm_backbone",
        "dinov2",
        "clip_backbones",
        "dofa",
        "croma",
    ):
        try:
            __import__(f"{__package__}.{mod}", fromlist=["*"])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not import backbone module %r: %s", mod, exc)


def _filter_kwargs(klass: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by ``klass.__init__``.

    If the constructor accepts ``**kwargs`` (a ``VAR_KEYWORD`` parameter), all
    kwargs are passed through unchanged. On any introspection failure the kwargs
    are returned as-is (the construction ``try/except`` still guards us).
    """
    try:
        sig = inspect.signature(klass.__init__)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return dict(kwargs)
    params = sig.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)
    allowed = {
        p.name
        for p in params
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.name != "self"
    }
    return {k: v for k, v in kwargs.items() if k in allowed}


def get_backbone(name: str, **kwargs: Any) -> Backbone:
    """Construct a backbone by *name*, falling back to ``FallbackBackbone``.

    This is the single entry point the pipeline / config layer uses. It:

    1. ensures the concrete backbone modules are imported (self-registration);
    2. looks up *name* (resolving aliases, case-insensitive);
    3. instantiates the class with ``**kwargs``;
    4. on **any** failure during lookup or construction -- unknown name, a
       missing optional dependency (``torch``/``transformers``/...), or a
       weight-download error -- logs a warning and returns a
       :class:`FallbackBackbone` so retrieval can still run offline.

    Parameters
    ----------
    name:
        Registry key or alias (e.g. ``"dofa"``, ``"openclip"``, ``"hashfeat"``).
    **kwargs:
        Forwarded to the backbone constructor (e.g. ``embed_dim=256``,
        ``model_name=...``, ``device="cpu"``). Because different backbones have
        different signatures, kwargs the target constructor does not accept are
        **silently dropped** (unless the constructor takes ``**kwargs``). This
        lets a shared config dict (e.g. a global ``image_size`` / ``device``) be
        passed to any backbone without spuriously triggering the fallback.

    Returns
    -------
    Backbone
        The requested backbone, or a fallback instance on failure.
    """
    _import_concrete_backbones()
    key = _resolve_name(name)
    klass = REGISTRY.get(key)

    if klass is None:
        logger.warning(
            "Unknown backbone %r (resolved %r); known: %s. Falling back to "
            "'fallback'.",
            name,
            key,
            sorted(REGISTRY),
        )
        return _make_fallback(name, **kwargs)

    try:
        return klass(**_filter_kwargs(klass, kwargs))
    except Exception as exc:
        logger.warning(
            "Failed to construct backbone %r (%s: %s). Falling back to "
            "offline 'fallback' backbone.",
            name,
            type(exc).__name__,
            exc,
        )
        return _make_fallback(name, **kwargs)


def _make_fallback(requested: str, **kwargs: Any) -> Backbone:
    """Build the offline fallback, forwarding only kwargs it understands."""
    from .fallback import FallbackBackbone

    fb_kwargs: dict[str, Any] = {}
    if "embed_dim" in kwargs:
        fb_kwargs["embed_dim"] = kwargs["embed_dim"]
    if "device" in kwargs:
        fb_kwargs["device"] = kwargs["device"]
    if "image_size" in kwargs:
        fb_kwargs["image_size"] = kwargs["image_size"]
    fb = FallbackBackbone(**fb_kwargs)
    fb._requested_name = requested  # type: ignore[attr-defined]
    return fb
