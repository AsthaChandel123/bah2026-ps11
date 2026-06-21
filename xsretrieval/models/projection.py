"""Projection heads and parameter-efficient adapters (torch).

This module provides the *trainable* alignment layer that sits on top of a
frozen backbone to produce the shared cross-modal embedding space:

* :class:`ProjectionHeads` -- per-modality 2-layer MLP
  (``Linear -> GELU -> Dropout -> Linear``) whose **final linear layer is
  weight-shared across modalities** when ``share_final=True``. Sharing the last
  layer biases all modalities through one common function, which is key to a
  single comparable space and reduces the modality gap (see research §2/§13).
  ``forward(x, modality)`` returns an L2-normalized ``(B, out_dim)`` embedding.
* :class:`LoRALinear` -- a low-rank (``r=8``) adapter usable to wrap a backbone
  ``nn.Linear`` for parameter-efficient fine-tuning (LoRA, Hu et al. 2022).
* :func:`freeze` / :func:`unfreeze` -- toggle ``requires_grad`` on a module.

Lazy-torch pattern
-------------------
``torch`` is **not** imported at module top level, so
``import xsretrieval.models.projection`` succeeds on a bare-numpy install. The
torch-backed classes (:class:`ProjectionHeads`, :class:`LoRALinear`) are built
lazily on first attribute access via :pep:`562` module ``__getattr__``: the
class objects are real ``torch.nn.Module`` subclasses (so ``isinstance`` and
subclassing work), but they only come into existence -- and only require torch
-- when you actually reference them. Constructing them without torch raises a
clear ``ImportError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

# ``ProjectionHeads`` / ``LoRALinear`` are bound lazily via module ``__getattr__``
# (PEP 562) and so are invisible to static analysis -- hence the noqa.
__all__ = ["ProjectionHeads", "LoRALinear", "freeze", "unfreeze"]  # noqa: F822

# Cache of lazily-built classes so each is constructed once.
_CLASS_CACHE: dict[str, type] = {}


# ---------------------------------------------------------------------------
# torch-free utilities (work as long as the passed module is a torch module)
# ---------------------------------------------------------------------------
def freeze(module: "nn.Module", *, requires_grad: bool = False) -> "nn.Module":
    """Set ``requires_grad`` (default ``False``) on all params of *module*.

    Returns the module for chaining. Does not import torch itself -- it just
    iterates ``module.parameters()``.
    """
    for p in module.parameters():
        p.requires_grad_(requires_grad)
    return module


def unfreeze(module: "nn.Module") -> "nn.Module":
    """Enable gradients on all parameters of *module* (inverse of :func:`freeze`)."""
    return freeze(module, requires_grad=True)


def _modality_key(modality: "Modality | str | None") -> str:
    """Stable string key for a modality (mirrors the backbone helper)."""
    if modality is None:
        return "unknown"
    value = getattr(modality, "value", None)
    if isinstance(value, str):
        return value.lower()
    name = getattr(modality, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(modality).lower()


# ---------------------------------------------------------------------------
# Lazy class builders
# ---------------------------------------------------------------------------
def _build_projection_heads_cls() -> type:
    from torch import nn  # local, lazy

    class ProjectionHeads(nn.Module):
        """Per-modality MLP projection heads with an optional shared final layer.

        Architecture (per modality): ``Linear(in_dim, hidden) -> GELU ->
        Dropout(dropout) -> Linear(hidden, out_dim)``. When ``share_final`` is
        ``True``, the second ``Linear(hidden, out_dim)`` is a *single* module
        shared across all modalities; only the first layer is modality-specific.
        The output is L2-normalized.

        Parameters
        ----------
        in_dim:
            Input feature width (the backbone's ``embed_dim``).
        out_dim:
            Output embedding width. Default ``256``.
        modalities:
            Iterable of modalities (enum members or strings) to create heads
            for. Each becomes a key; ``forward`` selects by the same key.
        share_final:
            Share the final linear layer across modalities. Default ``True``.
        hidden:
            Hidden width of the MLP. Default ``512``.
        dropout:
            Dropout probability after the GELU. Default ``0.1``.
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int = 256,
            modalities: Sequence["Modality | str"] | None = None,
            share_final: bool = True,
            hidden: int = 512,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.hidden = int(hidden)
            self.share_final = bool(share_final)
            mods = list(modalities) if modalities is not None else ["unknown"]
            self.modality_keys = [_modality_key(m) for m in mods]

            # First (modality-specific) layer per modality.
            self.first = nn.ModuleDict(
                {k: nn.Linear(self.in_dim, self.hidden) for k in self.modality_keys}
            )
            self.act = nn.GELU()
            self.dropout = nn.Dropout(float(dropout))

            # Final layer: shared single module, or one per modality.
            if self.share_final:
                self.shared_final = nn.Linear(self.hidden, self.out_dim)
                self.final = None
            else:
                self.shared_final = None
                self.final = nn.ModuleDict(
                    {
                        k: nn.Linear(self.hidden, self.out_dim)
                        for k in self.modality_keys
                    }
                )

        def _key(self, modality: "Modality | str | None") -> str:
            key = _modality_key(modality)
            if key in self.first:
                return key
            # Unknown modality at call time: fall back to the first registered
            # head so the module still produces an embedding.
            return self.modality_keys[0]

        def forward(
            self, x: "torch.Tensor", modality: "Modality | str | None" = None
        ) -> "torch.Tensor":
            """Project *x* for *modality* -> L2-normalized ``(B, out_dim)``."""
            key = self._key(modality)
            h = self.first[key](x)
            h = self.dropout(self.act(h))
            if self.share_final:
                z = self.shared_final(h)
            else:
                z = self.final[key](h)
            return nn.functional.normalize(z, p=2.0, dim=-1)

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"hidden={self.hidden}, share_final={self.share_final}, "
                f"modalities={self.modality_keys}"
            )

    return ProjectionHeads


def _build_lora_linear_cls() -> type:
    import math

    import torch  # local, lazy
    from torch import nn

    class LoRALinear(nn.Module):
        """Low-rank adapter (LoRA) around a frozen ``nn.Linear``.

        Computes ``W0 x + b0 + (alpha / r) * B(A x)`` where ``A`` (``r x in``)
        and ``B`` (``out x r``) are the only trainable parameters; the wrapped
        base linear ``(W0, b0)`` is frozen. ``B`` is zero-initialized so the
        adapter is an identity at the start of training. After training the
        update can be merged into ``W0`` (:meth:`merge`) for zero inference
        overhead.

        Parameters
        ----------
        base:
            The frozen ``nn.Linear`` to adapt. Its parameters are frozen here.
        r:
            LoRA rank. Default ``8``.
        alpha:
            LoRA scaling numerator (effective scale ``alpha / r``). Default
            equal to ``r`` (scale 1.0).
        dropout:
            Dropout applied to the LoRA input. Default ``0.0``.

        Examples
        --------
        Wrap the query projection of a transformer block::

            block.attn.qkv = LoRALinear(block.attn.qkv, r=8)
        """

        def __init__(
            self,
            base: "nn.Linear",
            r: int = 8,
            alpha: float | None = None,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError("LoRALinear expects an nn.Linear as `base`")
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.r = int(r)
            self.alpha = float(alpha if alpha is not None else r)
            self.scaling = self.alpha / max(self.r, 1)
            in_features = base.in_features
            out_features = base.out_features
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
            self.lora_dropout = (
                nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
            )
            # Kaiming init for A, zeros for B -> zero initial update.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.base(x)
            lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            return out + self.scaling * lora

        @torch.no_grad()
        def merge(self) -> "nn.Linear":
            """Fold the LoRA update into a standalone ``nn.Linear`` and return it.

            Produces a plain linear with ``W = W0 + scaling * B @ A`` so the
            adapter incurs zero inference cost.
            """
            merged = nn.Linear(
                self.base.in_features,
                self.base.out_features,
                bias=self.base.bias is not None,
            )
            delta = self.scaling * (self.lora_B @ self.lora_A)
            merged.weight.copy_(self.base.weight + delta)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
            return merged

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.r}, alpha={self.alpha}"
            )

    return LoRALinear


_BUILDERS = {
    "ProjectionHeads": _build_projection_heads_cls,
    "LoRALinear": _build_lora_linear_cls,
}


def __getattr__(name: str) -> Any:  # PEP 562 module-level lazy attributes
    """Lazily build and cache the torch-backed classes on first access.

    This is what lets ``import xsretrieval.models.projection`` succeed without
    torch while ``projection.ProjectionHeads`` / ``projection.LoRALinear`` are
    genuine ``torch.nn.Module`` subclasses once referenced.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    if name not in _CLASS_CACHE:
        try:
            _CLASS_CACHE[name] = builder()
        except ImportError as exc:  # torch missing
            raise ImportError(
                f"{name} requires PyTorch, which is not installed. "
                "Install torch to construct projection heads / adapters."
            ) from exc
    return _CLASS_CACHE[name]


def __dir__() -> list[str]:  # pragma: no cover - cosmetic
    return sorted(set(__all__) | set(globals()))
