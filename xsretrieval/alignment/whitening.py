"""Per-modality mean-centering + PCA whitening — the modality-gap fix.

This module implements the **single highest-ROI cross-modal alignment trick**
for the BAH 2026 PS-11 retrieval challenge (research note
``03_crossmodal_alignment.md`` §13, *GR-CLIP / BERT-whitening*).  It is **pure
numpy** (only ``numpy.linalg`` / SVD), so it runs before torch or faiss are ever
installed and forms the backbone of the "zero-training fallback" pipeline
(research §15).

Background — the modality gap
-----------------------------
Contrastive / foundation encoders place each sensor modality in its own narrow
*cone* on the unit hypersphere, separated from the other modalities by a roughly
**constant offset** (Liang et al., *Mind the Gap*, NeurIPS 2022).  Even when an
optical-forest and a SAR-forest are *semantically* the same scene, their cosine
similarity is depressed below that of two unrelated same-modality images, so a
naive shared index returns mostly same-modality neighbours and cross-modal F1
collapses.

The fix has three cumulative layers, applied **per modality**:

1. **Mean-centering** ``e' = e - mu_m`` (GR-CLIP).  Removes the constant offset
   between cones — the largest single win (up to +26 NDCG@10 in GR-CLIP).
2. **PCA whitening** ``e'' = W_m e'`` with ``W_m = V diag(1/sqrt(lambda)) V^T``
   (or its dimensionality-reducing variant).  Makes each modality's cloud
   *isotropic* so cosine similarity becomes a faithful semantic measure and
   bursty / anisotropic directions stop dominating (Mahalanobis equivalence).
3. **Top principal-component removal** (optional).  The dominant principal
   direction(s) frequently encode *sensor identity / global energy* rather than
   land-cover semantics; dropping them further closes the gap (akin to removing
   the top PCs in BERT-whitening / "all-but-the-top").

Finally we **L2-normalize** so downstream cosine == inner product (faiss
``METRIC_INNER_PRODUCT``).

Mathematical detail
-------------------
Given centred data ``X`` (n, d) for a modality, the covariance is
``C = X^T X / (n - 1)``.  Its eigendecomposition ``C = V Λ V^T`` (``V``
orthonormal, ``Λ`` diagonal, eigenvalues sorted descending) gives the PCA
whitening transform

    W = V_keep @ diag(1 / sqrt(λ_keep + eps)) @ V_keep^T          (ZCA-style)

or, when reducing to ``k`` components,

    W = diag(1 / sqrt(λ_1:k + eps)) @ V_1:k^T   (shape (k, d), PCA-style)

We compute the eigendecomposition from the **SVD of the centred data**
(``X = U S V^T`` ⇒ eigenvalues ``λ_i = S_i^2 / (n-1)``), which is numerically
more stable than forming ``C`` explicitly.  Optional **shrinkage** regularises
the covariance toward the identity (Ledoit-Wolf style),
``λ <- (1-α) λ + α * mean(λ)``, which is important on small reference sets where
the empirical covariance is noisy and over-whitening hurts.

Complexity
----------
``fit`` is ``O(n d^2)`` (thin SVD) per modality, ``transform`` is ``O(n d^2)``
for the dense matrix multiply (``O(n d k)`` when reducing to ``k`` dims).  Both
are one-off, cheap (d is 256-768, n a few thousand) and CPU-only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from xsretrieval.data.modalities import Modality

__all__ = [
    "PerModalityWhitener",
    "GlobalWhitener",
    "mean_center_per_modality",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_modality(modality) -> Modality:
    """Coerce ``modality`` (enum, raw string, or ``Modality``) to ``Modality``."""
    if isinstance(modality, Modality):
        return modality
    return Modality(modality)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; rows with ~0 norm are left at 0."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


def _fit_whitening(
    emb: np.ndarray,
    *,
    n_components: Optional[int],
    remove_top_pc: int,
    shrinkage: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a mean + whitening matrix for a single modality's embeddings.

    Returns ``(mean, W)`` where ``mean`` has shape ``(d,)`` and ``W`` has shape
    ``(out_dim, d)`` so that ``z = (e - mean) @ W.T`` is the whitened (and
    optionally dimensionality-reduced / top-PC-removed) embedding.

    Parameters mirror :class:`PerModalityWhitener`.
    """
    emb = np.asarray(emb, dtype=np.float64)  # float64 for a stable SVD
    if emb.ndim != 2:
        raise ValueError(f"expected (n, d) embeddings, got shape {emb.shape}")
    n, d = emb.shape

    mean = emb.mean(axis=0)
    xc = emb - mean

    if n < 2:
        # Cannot estimate a covariance from a single point: identity transform.
        W = np.eye(d, dtype=np.float32)
        return mean.astype(np.float32), W

    # SVD of the centred data: xc = U S Vt. Eigenvalues of cov = S^2 / (n-1).
    # full_matrices=False gives the thin SVD (min(n, d) singular values).
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    eigvals = (s ** 2) / (n - 1)  # descending order
    # Pad to d components if the thin SVD returned fewer (n-1 < d case).
    if eigvals.shape[0] < d:
        pad = d - eigvals.shape[0]
        eigvals = np.concatenate([eigvals, np.zeros(pad, dtype=eigvals.dtype)])
        vt = np.concatenate(
            [vt, np.zeros((pad, d), dtype=vt.dtype)], axis=0
        )

    # Optional shrinkage toward the identity (Ledoit-Wolf style): pulls the
    # eigenvalue spectrum toward its mean, damping over-whitening of noisy
    # low-variance directions on small reference sets.
    if shrinkage and shrinkage > 0.0:
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
        mu_eig = float(eigvals.mean())
        eigvals = (1.0 - alpha) * eigvals + alpha * mu_eig

    # Components to keep: drop the top ``remove_top_pc`` (sensor-identity
    # directions), then keep ``n_components`` of the remainder.
    start = max(0, int(remove_top_pc))
    total = vt.shape[0]
    if start >= total:
        raise ValueError(
            f"remove_top_pc={remove_top_pc} removes all {total} components"
        )
    end = total
    if n_components is not None:
        end = min(total, start + int(n_components))

    keep_vt = vt[start:end]                 # (k, d)
    keep_eig = eigvals[start:end]           # (k,)
    inv_sqrt = 1.0 / np.sqrt(keep_eig + eps)  # (k,)

    # PCA-style whitening matrix W (k, d): z = (e - mean) @ W.T.
    # Row i of W is inv_sqrt[i] * V[:, i] = inv_sqrt[i] * keep_vt[i].
    W = (keep_vt * inv_sqrt[:, None]).astype(np.float32)
    return mean.astype(np.float32), W


class PerModalityWhitener:
    """Per-modality mean-center + PCA-whitening (+ optional top-PC removal).

    Fits an independent ``(mean_m, W_m)`` pair for every modality so that

        z = L2_normalize( (e - mean_m) @ W_m.T )

    lives in an isotropic, gap-corrected shared space.  This is the recommended
    inference-time transform for both the trained pipeline and the zero-shot
    fallback (research ``03_crossmodal_alignment.md`` §14-15).

    Parameters
    ----------
    n_components:
        Number of principal components to keep *after* removing the top
        ``remove_top_pc`` (i.e. the output dimensionality of the whitened
        embedding).  ``None`` keeps all available components (full whitening,
        output dim == input dim minus ``remove_top_pc``).
    remove_top_pc:
        Number of leading principal components to discard before whitening.
        These often encode sensor identity / global energy rather than
        semantics; dropping 1-2 can further shrink the modality gap.  Default
        ``0`` (keep everything).
    shrinkage:
        Covariance shrinkage coefficient in ``[0, 1]`` (Ledoit-Wolf style).
        ``0.0`` = pure empirical whitening; small positive values (e.g. 0.1)
        stabilise whitening on small reference sets.
    eps:
        Numerical floor added to eigenvalues before the inverse square root.

    Notes
    -----
    * The transform for an **unseen** modality (not present at ``fit`` time)
      falls back to mean-centering with the *global* mean and L2-normalisation
      only, so the engine never crashes on a novel sensor — it simply forgoes
      whitening for that modality.
    * All state is plain numpy; :meth:`save` / :meth:`load` use ``np.savez``.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        remove_top_pc: int = 0,
        shrinkage: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        if remove_top_pc < 0:
            raise ValueError("remove_top_pc must be >= 0")
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be a positive integer or None")
        self.n_components = n_components
        self.remove_top_pc = int(remove_top_pc)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)

        # Fitted state (keyed by Modality value string for serialisability).
        self.means_: dict[str, np.ndarray] = {}
        self.transforms_: dict[str, np.ndarray] = {}
        self.global_mean_: Optional[np.ndarray] = None
        self.input_dim_: Optional[int] = None
        self.fitted_: bool = False

    # -- fitting ----------------------------------------------------------
    def fit(self, emb_by_mod: dict) -> "PerModalityWhitener":
        """Fit per-modality mean + whitening transforms.

        Parameters
        ----------
        emb_by_mod:
            Mapping ``{Modality (or its str value): np.ndarray (n_m, d)}`` of
            reference embeddings per modality.  Use a *representative* sample of
            the gallery (a few thousand vectors per modality is plenty).

        Returns
        -------
        self
        """
        if not emb_by_mod:
            raise ValueError("emb_by_mod is empty; nothing to fit")

        self.means_.clear()
        self.transforms_.clear()

        all_rows: list[np.ndarray] = []
        dim: Optional[int] = None
        for modality, emb in emb_by_mod.items():
            mod = _as_modality(modality)
            arr = np.asarray(emb, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"embeddings for {mod} must be (n, d); got {arr.shape}"
                )
            if arr.shape[0] == 0:
                continue
            if dim is None:
                dim = arr.shape[1]
            elif arr.shape[1] != dim:
                raise ValueError(
                    f"inconsistent embedding dim for {mod}: "
                    f"{arr.shape[1]} != {dim}"
                )
            mean, W = _fit_whitening(
                arr,
                n_components=self.n_components,
                remove_top_pc=self.remove_top_pc,
                shrinkage=self.shrinkage,
                eps=self.eps,
            )
            self.means_[mod.value] = mean
            self.transforms_[mod.value] = W
            all_rows.append(arr)

        if dim is None:
            raise ValueError("all modalities had zero embeddings; cannot fit")

        self.input_dim_ = int(dim)
        self.global_mean_ = (
            np.concatenate(all_rows, axis=0).mean(axis=0).astype(np.float32)
        )
        self.fitted_ = True
        return self

    # -- transforming -----------------------------------------------------
    def transform(self, emb: np.ndarray, modality) -> np.ndarray:
        """Apply mean-center → whiten → (top-PC removal) → L2-normalize.

        Parameters
        ----------
        emb:
            Embeddings ``(n, d)`` of a *single* modality.
        modality:
            The modality of every row in ``emb``.

        Returns
        -------
        np.ndarray
            Float32 ``(n, out_dim)`` whitened, L2-normalized embeddings.
        """
        if not self.fitted_:
            raise RuntimeError("PerModalityWhitener.transform before fit()")
        arr = np.asarray(emb, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"expected (n, d) embeddings, got {arr.shape}")

        mod = _as_modality(modality)
        key = mod.value
        if key in self.transforms_:
            mean = self.means_[key]
            W = self.transforms_[key]
            out = (arr - mean) @ W.T
        else:
            # Unseen modality: fall back to global mean-centering only.
            mean = self.global_mean_
            if mean is None:
                mean = arr.mean(axis=0)
            out = arr - mean
        out = _l2_normalize(out)
        return out[0] if single else out

    def fit_transform(self, emb_by_mod: dict) -> dict:
        """Fit on ``emb_by_mod`` then transform each modality's embeddings.

        Returns
        -------
        dict
            ``{Modality: np.ndarray}`` of whitened embeddings, keyed by the same
            ``Modality`` enum members passed in.
        """
        self.fit(emb_by_mod)
        out: dict = {}
        for modality, emb in emb_by_mod.items():
            mod = _as_modality(modality)
            arr = np.asarray(emb, dtype=np.float32)
            if arr.shape[0] == 0:
                out[mod] = arr
            else:
                out[mod] = self.transform(arr, mod)
        return out

    # -- (de)serialisation ------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise the fitted transforms to a ``.npz`` file."""
        if not self.fitted_:
            raise RuntimeError("cannot save an unfitted PerModalityWhitener")
        payload: dict[str, np.ndarray] = {
            "__config__": np.array(
                [
                    -1 if self.n_components is None else self.n_components,
                    self.remove_top_pc,
                ],
                dtype=np.int64,
            ),
            "__floatcfg__": np.array(
                [self.shrinkage, self.eps], dtype=np.float64
            ),
            "__modalities__": np.array(
                list(self.transforms_.keys()), dtype=object
            ),
            "__input_dim__": np.array([self.input_dim_], dtype=np.int64),
            "__global_mean__": self.global_mean_,
        }
        for key in self.transforms_:
            payload[f"mean::{key}"] = self.means_[key]
            payload[f"W::{key}"] = self.transforms_[key]
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "PerModalityWhitener":
        """Load a fitted :class:`PerModalityWhitener` from a ``.npz`` file."""
        data = np.load(path, allow_pickle=True)
        cfg = data["__config__"]
        n_components = None if int(cfg[0]) < 0 else int(cfg[0])
        remove_top_pc = int(cfg[1])
        fcfg = data["__floatcfg__"]
        obj = cls(
            n_components=n_components,
            remove_top_pc=remove_top_pc,
            shrinkage=float(fcfg[0]),
            eps=float(fcfg[1]),
        )
        keys = [str(k) for k in data["__modalities__"].tolist()]
        for key in keys:
            obj.means_[key] = np.asarray(data[f"mean::{key}"], dtype=np.float32)
            obj.transforms_[key] = np.asarray(data[f"W::{key}"], dtype=np.float32)
        obj.input_dim_ = int(data["__input_dim__"][0])
        gm = data["__global_mean__"]
        obj.global_mean_ = None if gm is None else np.asarray(gm, dtype=np.float32)
        obj.fitted_ = True
        return obj


class GlobalWhitener:
    """Single (modality-agnostic) mean-center + PCA-whitening transform.

    Identical mathematics to :class:`PerModalityWhitener` but fits **one**
    ``(mean, W)`` on the union of all embeddings.  Useful when (a) you train one
    shared backbone whose modality gap is already small, or (b) you want a plain
    descriptor-whitening / dimensionality-reduction stage (research
    ``05_training_losses.md`` §0.3 — PCA-whitening as post-processing).

    Parameters are the same as :class:`PerModalityWhitener`.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        remove_top_pc: int = 0,
        shrinkage: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        if remove_top_pc < 0:
            raise ValueError("remove_top_pc must be >= 0")
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be a positive integer or None")
        self.n_components = n_components
        self.remove_top_pc = int(remove_top_pc)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    def fit(self, emb: np.ndarray) -> "GlobalWhitener":
        """Fit the global mean + whitening matrix on ``emb`` (n, d)."""
        mean, W = _fit_whitening(
            np.asarray(emb, dtype=np.float32),
            n_components=self.n_components,
            remove_top_pc=self.remove_top_pc,
            shrinkage=self.shrinkage,
            eps=self.eps,
        )
        self.mean_ = mean
        self.W_ = W
        self.fitted_ = True
        return self

    def transform(self, emb: np.ndarray, modality=None) -> np.ndarray:
        """Whiten ``emb`` and L2-normalize.

        ``modality`` is accepted (and ignored) so :class:`GlobalWhitener` is a
        drop-in replacement for :class:`PerModalityWhitener` in the engine.
        """
        if not self.fitted_:
            raise RuntimeError("GlobalWhitener.transform before fit()")
        arr = np.asarray(emb, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        out = (arr - self.mean_) @ self.W_.T
        out = _l2_normalize(out)
        return out[0] if single else out

    def fit_transform(self, emb: np.ndarray) -> np.ndarray:
        """Fit on ``emb`` then return the whitened, L2-normalized embeddings."""
        return self.fit(emb).transform(emb)

    def save(self, path: str) -> None:
        """Serialise to a ``.npz`` file."""
        if not self.fitted_:
            raise RuntimeError("cannot save an unfitted GlobalWhitener")
        np.savez(
            path,
            mean=self.mean_,
            W=self.W_,
            config=np.array(
                [
                    -1 if self.n_components is None else self.n_components,
                    self.remove_top_pc,
                ],
                dtype=np.int64,
            ),
            floatcfg=np.array([self.shrinkage, self.eps], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str) -> "GlobalWhitener":
        """Load a fitted :class:`GlobalWhitener` from a ``.npz`` file."""
        data = np.load(path, allow_pickle=True)
        cfg = data["config"]
        fcfg = data["floatcfg"]
        obj = cls(
            n_components=None if int(cfg[0]) < 0 else int(cfg[0]),
            remove_top_pc=int(cfg[1]),
            shrinkage=float(fcfg[0]),
            eps=float(fcfg[1]),
        )
        obj.mean_ = np.asarray(data["mean"], dtype=np.float32)
        obj.W_ = np.asarray(data["W"], dtype=np.float32)
        obj.fitted_ = True
        return obj


def mean_center_per_modality(
    emb: np.ndarray,
    modalities,
    *,
    means: Optional[dict] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Subtract each row's per-modality mean (GR-CLIP), then L2-normalize.

    This is the cheapest possible modality-gap remedy (research §13 remedy 1):
    it removes the constant offset between modality cones with *zero* training
    and no covariance estimation.

    Parameters
    ----------
    emb:
        Embeddings ``(n, d)``.
    modalities:
        Length-``n`` sequence / array of ``Modality`` (or their string values),
        one per row of ``emb``.
    means:
        Optional precomputed ``{Modality (or str): mean (d,)}`` to subtract
        (e.g. fitted on the full gallery and reused for queries).  If ``None``,
        per-modality means are computed from ``emb`` itself.
    normalize:
        Whether to L2-normalize the result (default ``True``).

    Returns
    -------
    np.ndarray
        Float32 ``(n, d)`` mean-centred (and optionally normalized) embeddings.
    """
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected (n, d) embeddings, got {arr.shape}")
    mods = [_as_modality(m) for m in modalities]
    if len(mods) != arr.shape[0]:
        raise ValueError(
            f"modalities length {len(mods)} != n rows {arr.shape[0]}"
        )

    out = arr.copy()
    keys = np.array([m.value for m in mods])
    if means is None:
        # Compute per-modality means from the data itself.
        for key in np.unique(keys):
            mask = keys == key
            out[mask] -= arr[mask].mean(axis=0)
    else:
        norm_means = {_as_modality(k).value: np.asarray(v, dtype=np.float32)
                      for k, v in means.items()}
        for key in np.unique(keys):
            mask = keys == key
            if key in norm_means:
                out[mask] -= norm_means[key]
            else:
                out[mask] -= arr[mask].mean(axis=0)

    if normalize:
        out = _l2_normalize(out)
    return out
