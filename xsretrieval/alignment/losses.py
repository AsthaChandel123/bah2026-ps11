"""Cross-modal training losses for the shared embedding space.

Implements the loss recipe recommended in research note
``05_training_losses.md`` §13 for BAH 2026 PS-11:

    L = 1.5 * symmetric_infonce        # closes the modality gap (CLIP-style)
      + 1.0 * SubCenterArcFace         # tight, label-noise-robust class clusters
      + 0.5 * batch_hard_triplet       # sharpens cross-modal top-K rank order

plus optional ranking-direct losses (:func:`multi_similarity_loss`,
:func:`smooth_ap`) that target F1@K more directly, used as a final-stage polish.

torch policy
------------
This module performs **no torch import at module load time** so that
``import xsretrieval.alignment.losses`` works in a numpy-only environment (e.g.
to inspect the public API, or before the training extras are installed).  Every
function imports torch lazily *inside its body*, and every ``nn.Module`` subclass
is built by a factory (:func:`SubCenterArcFace`, :func:`CrossModalRetrievalLoss`)
that imports torch at construction time and returns a real ``nn.Module``
instance.  Calling any of these without torch installed raises a clear
``ImportError``.

All embeddings are assumed **L2-normalized** (unit hypersphere); the angular /
cosine maths below relies on it.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "symmetric_infonce",
    "info_nce",
    "SubCenterArcFace",
    "batch_hard_triplet",
    "multi_similarity_loss",
    "smooth_ap",
    "CrossModalRetrievalLoss",
]


def _torch():
    """Lazily import and return the ``torch`` module (clear error if missing)."""
    try:
        import torch  # noqa: F401  (imported for side effect of availability)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xsretrieval.alignment.losses requires PyTorch. Install the "
            "training extras (e.g. `pip install torch`) to use the loss "
            "functions; the numpy retrieval path does not need them."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Contrastive (InfoNCE) losses
# ---------------------------------------------------------------------------
def symmetric_infonce(z_a, z_b, temperature: float = 0.07):
    r"""CLIP-style **symmetric** cross-modal InfoNCE (NT-Xent).

    For a batch of ``N`` aligned pairs with L2-normalized embeddings
    ``u_i = z_a[i]`` (modality A) and ``v_i = z_b[i]`` (modality B), and
    temperature ``tau``:

    .. math::

        L_{a\to b} = -\frac1N \sum_i \log
            \frac{\exp(u_i\cdot v_i/\tau)}{\sum_j \exp(u_i\cdot v_j/\tau)}

        L_{b\to a} = -\frac1N \sum_i \log
            \frac{\exp(v_i\cdot u_i/\tau)}{\sum_j \exp(v_i\cdot u_j/\tau)}

        L = \tfrac12 (L_{a\to b} + L_{b\to a})

    The matched pair ``(u_i, v_i)`` is the positive; all other in-batch pairs are
    negatives.  This is the bilateral generalisation of CLIP from (image, text)
    to (sensorA, sensorB) and is the term that forces a *shared* space rather
    than a one-directional projection.

    Parameters
    ----------
    z_a, z_b:
        ``(N, D)`` tensors of L2-normalized embeddings, paired row-wise.
    temperature:
        Softmax temperature ``tau`` (CLIP default 0.07). Lower ``tau`` emphasises
        hard negatives (sharper, but widens the modality gap, research §13).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    torch = _torch()
    import torch.nn.functional as F

    if z_a.shape != z_b.shape:
        raise ValueError(f"z_a {tuple(z_a.shape)} != z_b {tuple(z_b.shape)}")
    n = z_a.shape[0]
    # Logits matrix S_ij = (z_a[i] . z_b[j]) / tau ; diagonal are the positives.
    logits = (z_a @ z_b.t()) / float(temperature)
    targets = torch.arange(n, device=z_a.device)
    loss_a2b = F.cross_entropy(logits, targets)
    loss_b2a = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a2b + loss_b2a)


def info_nce(z, z_pos, z_negs=None, temperature: float = 0.07):
    r"""One-directional InfoNCE for anchors ``z`` and positives ``z_pos``.

    .. math::

        L = -\frac1N \sum_i \log
            \frac{\exp(z_i\cdot z^+_i/\tau)}
                 {\exp(z_i\cdot z^+_i/\tau) + \sum_k \exp(z_i\cdot n_k/\tau)}

    Negatives are, by default, **all other positives in the batch** (the
    standard in-batch-negatives formulation).  An explicit negative bank
    ``z_negs`` (e.g. a MoCo queue, research ``05_training_losses.md`` §10) may be
    supplied instead/in addition.

    Parameters
    ----------
    z:
        ``(N, D)`` anchor embeddings (L2-normalized).
    z_pos:
        ``(N, D)`` positive embeddings, paired row-wise with ``z``.
    z_negs:
        Optional ``(M, D)`` explicit negatives. When ``None`` the other rows of
        ``z_pos`` act as negatives (in-batch negatives).
    temperature:
        Softmax temperature ``tau``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    torch = _torch()
    import torch.nn.functional as F

    if z.shape != z_pos.shape:
        raise ValueError(f"z {tuple(z.shape)} != z_pos {tuple(z_pos.shape)}")
    n = z.shape[0]
    tau = float(temperature)

    if z_negs is None:
        # In-batch negatives: logits against all positives, diagonal = positive.
        logits = (z @ z_pos.t()) / tau
        targets = torch.arange(n, device=z.device)
        return F.cross_entropy(logits, targets)

    # Explicit negative bank: positive logit is the paired dot product, negatives
    # are shared across all anchors. Concatenate [pos | negs] and target idx 0.
    pos_logit = (z * z_pos).sum(dim=1, keepdim=True) / tau    # (N, 1)
    neg_logits = (z @ z_negs.t()) / tau                       # (N, M)
    logits = torch.cat([pos_logit, neg_logits], dim=1)        # (N, 1 + M)
    targets = torch.zeros(n, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Angular-margin classification loss (Sub-center ArcFace)
# ---------------------------------------------------------------------------
def SubCenterArcFace(
    in_dim: int,
    n_classes: int,
    K: int = 3,
    s: float = 64.0,
    m: float = 0.5,
):
    r"""Construct a **Sub-center ArcFace** ``nn.Module`` (factory).

    Sub-center ArcFace (Deng et al., 2020) is an additive **angular-margin**
    classification head whose normalized class weights become land-cover class
    *prototypes* and whose input features become the retrieval embedding.  Each
    class owns ``K`` sub-centers; a sample only has to be close to its *nearest*
    sub-center, which makes the loss robust to the large intra-class / seasonal /
    cross-sensor appearance variation of remote-sensing classes (research
    ``05_training_losses.md`` §5).  Feeding all modalities through the **one**
    shared prototype matrix is itself an implicit cross-modal aligner.

    Maths (per sample with embedding ``z``, label ``y``):

    .. math::

        \cos\theta_{j} = \max_{k\in[K]} \hat W_{j,k}^\top \hat z, \quad
        \text{logit}_j =
            \begin{cases}
                s\cdot\cos(\theta_y + m) & j = y\\
                s\cdot\cos\theta_j       & j \neq y
            \end{cases}

    followed by cross-entropy.  ``\hat\cdot`` denotes L2-normalisation; ``s`` is
    the scale that re-inflates the bounded cosine so softmax can saturate, ``m``
    the additive angular margin (added to the *geodesic* angle of the true
    class).

    Parameters
    ----------
    in_dim:
        Embedding dimensionality ``D``.
    n_classes:
        Number of land-cover / land-use classes.
    K:
        Sub-centers per class (default 3).
    s:
        Logit scale (default 64.0).
    m:
        Additive angular margin in radians (default 0.5 ≈ 28.6°).

    Returns
    -------
    torch.nn.Module
        Module whose ``forward(embeddings, labels) -> scalar loss``.
    """
    torch = _torch()
    import math

    import torch.nn as nn
    import torch.nn.functional as F

    class _SubCenterArcFace(nn.Module):
        """See :func:`SubCenterArcFace`."""

        def __init__(self) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.n_classes = int(n_classes)
            self.K = int(K)
            self.s = float(s)
            self.m = float(m)
            # Class sub-center prototypes: (n_classes * K, in_dim).
            self.weight = nn.Parameter(
                torch.empty(self.n_classes * self.K, self.in_dim)
            )
            nn.init.xavier_uniform_(self.weight)
            # Precompute margin trig constants.
            self._cos_m = math.cos(self.m)
            self._sin_m = math.sin(self.m)
            # cos(pi - m): the threshold of the monotonic region. Beyond it we
            # use the (Taylor) fallback cos(theta) - sin(m)*m to keep the
            # function monotonically decreasing in theta (ArcFace `easy_margin`
            # off / "hard" variant).
            self._th = math.cos(math.pi - self.m)
            self._mm = math.sin(math.pi - self.m) * self.m

        def forward(self, embeddings, labels):
            # Normalize embeddings and all sub-center prototypes.
            z = F.normalize(embeddings, p=2, dim=1)
            w = F.normalize(self.weight, p=2, dim=1)
            # Cosine to every sub-center, then max over the K sub-centers/class.
            cos_all = z @ w.t()                                  # (B, C*K)
            cos_all = cos_all.view(-1, self.n_classes, self.K)
            cos = cos_all.max(dim=2).values                      # (B, C)
            cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

            labels = labels.long()
            sin = torch.sqrt(1.0 - cos * cos)
            # cos(theta + m) = cos theta cos m - sin theta sin m, applied only to
            # the ground-truth class column.
            phi = cos * self._cos_m - sin * self._sin_m
            # Monotonicity guard (hard margin): where cos <= cos(pi - m), the
            # naive phi would increase again, so use the linear fallback.
            phi = torch.where(cos > self._th, phi, cos - self._mm)

            one_hot = F.one_hot(labels, num_classes=self.n_classes).to(cos.dtype)
            logits = self.s * (one_hot * phi + (1.0 - one_hot) * cos)
            return F.cross_entropy(logits, labels)

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"in_dim={self.in_dim}, n_classes={self.n_classes}, "
                f"K={self.K}, s={self.s}, m={self.m}"
            )

    return _SubCenterArcFace()


# ---------------------------------------------------------------------------
# Metric / ranking losses
# ---------------------------------------------------------------------------
def _pairwise_cosine_dist(z):
    """Pairwise cosine *distance* matrix ``1 - z z^T`` for unit ``z`` (B, B)."""
    torch = _torch()
    sim = z @ z.t()
    sim = sim.clamp(-1.0, 1.0)
    return 1.0 - sim


def _modalities_to_codes(modalities, device):
    """Coerce a modality batch to a ``(B,)`` long tensor of integer codes.

    Accepts any of: a torch ``LongTensor`` / tensor of codes (used as-is), a
    numpy array, or a Python sequence of
    :class:`~xsretrieval.data.modalities.Modality` enums / raw strings / ints.
    Only the *equality structure* matters for the cross-modal masks, so distinct
    modality values are mapped to distinct integers (preserving any existing
    integer codes). This makes the triplet / combined losses robust to the
    natural ``list[Modality]`` produced by the dataset layer, not just
    pre-encoded tensors.
    """
    torch = _torch()
    if torch.is_tensor(modalities):
        return modalities.to(device=device, dtype=torch.long)

    # numpy array or generic sequence -> stable integer codes by value.
    seq = list(modalities)
    codes: list[int] = []
    mapping: dict = {}
    for m in seq:
        if isinstance(m, (int,)) or (hasattr(m, "__index__") and not isinstance(m, str)):
            key = int(m)
        else:
            key = getattr(m, "value", m)  # Modality enum -> its string value
            key = str(key)
        if key not in mapping:
            mapping[key] = len(mapping)
        codes.append(mapping[key])
    return torch.as_tensor(codes, dtype=torch.long, device=device)


def batch_hard_triplet(
    embeddings,
    labels,
    modalities=None,
    margin: float = 0.1,
    cross_modal: bool = True,
):
    r"""Batch-hard triplet loss with optional cross-modal mining.

    For every anchor in the batch, mine the **hardest positive** (the same-class
    sample that is *farthest* away) and the **hardest negative** (the
    different-class sample that is *closest*), then apply the margin ranking loss

    .. math::

        L = \frac1B \sum_a \big[ d(a, p_{\text{hard}}) - d(a, n_{\text{hard}})
            + \text{margin} \big]_+

    with ``d`` the cosine distance ``1 - cos`` (research ``05_training_losses.md``
    §4, Hermans et al. batch-hard).

    When ``cross_modal`` is ``True`` and ``modalities`` is provided, positives
    are restricted to **different**-modality same-class samples and negatives to
    **different**-modality different-class samples *where such candidates exist*
    (falling back to any-modality otherwise).  This directly hardens the
    inter-modal class boundary that cross-modal F1@K depends on.

    Parameters
    ----------
    embeddings:
        ``(B, D)`` L2-normalized embeddings.
    labels:
        ``(B,)`` integer class labels.
    modalities:
        Optional ``(B,)`` integer (or long) modality ids; required for the
        cross-modal mining preference.
    margin:
        Triplet margin (default 0.1, the research recipe value).
    cross_modal:
        Prefer cross-modal positives/negatives when ``modalities`` is given.

    Returns
    -------
    torch.Tensor
        Scalar loss (0 if no valid triplet exists in the batch).
    """
    torch = _torch()

    z = embeddings
    b = z.shape[0]
    device = z.device
    labels = labels.long()

    dist = _pairwise_cosine_dist(z)                       # (B, B) in [0, 2]
    lab_eq = labels[:, None] == labels[None, :]           # (B, B) same class
    eye = torch.eye(b, dtype=torch.bool, device=device)

    pos_mask = lab_eq & ~eye                              # same class, not self
    neg_mask = ~lab_eq                                    # different class

    if cross_modal and modalities is not None:
        mod = _modalities_to_codes(modalities, device)
        if mod.ndim > 1:
            mod = mod.view(-1)
        diff_mod = mod[:, None] != mod[None, :]           # (B, B) cross-modal
        # Prefer cross-modal positives where any exist for the anchor.
        cm_pos = pos_mask & diff_mod
        has_cm_pos = cm_pos.any(dim=1, keepdim=True)
        pos_mask = torch.where(has_cm_pos, cm_pos, pos_mask)
        # Prefer cross-modal negatives where any exist for the anchor.
        cm_neg = neg_mask & diff_mod
        has_cm_neg = cm_neg.any(dim=1, keepdim=True)
        neg_mask = torch.where(has_cm_neg, cm_neg, neg_mask)

    # Hardest positive: max distance among positives (-inf where none → ignore).
    neg_inf = torch.tensor(float("-inf"), device=device)
    pos_inf = torch.tensor(float("inf"), device=device)
    pos_d = torch.where(pos_mask, dist, neg_inf)
    hardest_pos = pos_d.max(dim=1).values                # (B,)
    neg_d = torch.where(neg_mask, dist, pos_inf)
    hardest_neg = neg_d.min(dim=1).values                # (B,)

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if valid.sum() == 0:
        return z.sum() * 0.0  # keep graph connected, value 0
    losses = torch.clamp(
        hardest_pos[valid] - hardest_neg[valid] + float(margin), min=0.0
    )
    return losses.mean()


def multi_similarity_loss(
    embeddings,
    labels,
    alpha: float = 2.0,
    beta: float = 50.0,
    base: float = 0.5,
    epsilon: float = 0.1,
):
    r"""Multi-Similarity (MS) loss with general pair weighting (Wang et al. 2019).

    Two-stage *mine then weight* over cosine similarities ``S_ij = z_i^\top z_j``:
    after hard-pair mining (keep positives harder than the hardest negative minus
    ``epsilon``, negatives easier than the hardest positive plus ``epsilon``),

    .. math::

        L = \frac1B \sum_i \Big[
            \frac1\alpha \log\big(1 + \sum_{k\in P_i}
                e^{-\alpha (S_{ik}-\lambda)}\big)
          + \frac1\beta  \log\big(1 + \sum_{k\in N_i}
                e^{ \beta  (S_{ik}-\lambda)}\big) \Big]

    with self-, positive- and negative-similarity weighting (research
    ``05_training_losses.md`` §7).  A strong large-batch alternative to the
    triplet term.

    Parameters
    ----------
    embeddings:
        ``(B, D)`` L2-normalized embeddings.
    labels:
        ``(B,)`` integer class labels.
    alpha, beta, base, epsilon:
        MS hyperparameters (pytorch-metric-learning defaults:
        ``alpha=2, beta=50, base=0.5``; ``epsilon`` is the mining slack 0.1).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    torch = _torch()

    z = embeddings
    b = z.shape[0]
    device = z.device
    labels = labels.long()
    lam = float(base)

    sim = (z @ z.t()).clamp(-1.0, 1.0)                   # (B, B)
    lab_eq = labels[:, None] == labels[None, :]
    eye = torch.eye(b, dtype=torch.bool, device=device)
    pos_mask = lab_eq & ~eye
    neg_mask = ~lab_eq

    losses = []
    for i in range(b):
        pos_sim = sim[i][pos_mask[i]]
        neg_sim = sim[i][neg_mask[i]]
        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue
        # Hard mining: positives below (max neg + eps); negatives above
        # (min pos - eps).
        max_neg = neg_sim.max()
        min_pos = pos_sim.min()
        pos_sel = pos_sim[pos_sim - float(epsilon) < max_neg]
        neg_sel = neg_sim[neg_sim + float(epsilon) > min_pos]
        if pos_sel.numel() == 0:
            pos_sel = pos_sim
        if neg_sel.numel() == 0:
            neg_sel = neg_sim
        pos_term = (1.0 / alpha) * torch.log1p(
            torch.exp(-alpha * (pos_sel - lam)).sum()
        )
        neg_term = (1.0 / beta) * torch.log1p(
            torch.exp(beta * (neg_sel - lam)).sum()
        )
        losses.append(pos_term + neg_term)

    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def smooth_ap(scores_or_emb, labels, tau: float = 0.01):
    r"""Smooth-AP loss — a differentiable surrogate for Average Precision.

    F1@K is rank-based, so directly optimising AP is the most metric-aligned
    objective (research ``05_training_losses.md`` §9; Brown et al., ECCV 2020).
    The non-differentiable rank indicator in AP is replaced by a sigmoid
    ``G(x; tau) = 1 / (1 + e^{-x/tau})`` applied to pairwise score differences
    ``D_ij = S_i - S_j``:

    .. math::

        \text{AP}_q \approx \frac1{|P_q|} \sum_{i\in P_q}
            \frac{1 + \sum_{j\in P_q} G(D_{ji})}
                 {1 + \sum_{j\in \Omega} G(D_{ji})},\qquad
        L = \frac1m \sum_q (1 - \text{AP}_q)

    Parameters
    ----------
    scores_or_emb:
        Either a precomputed ``(B, B)`` similarity matrix or a ``(B, D)`` matrix
        of L2-normalized embeddings (cosine similarity is then computed).
    labels:
        ``(B,)`` integer class labels; relevance = same label.
    tau:
        Sigmoid temperature (smaller = closer to the true step function;
        default 0.01).

    Returns
    -------
    torch.Tensor
        Scalar loss ``mean(1 - AP)``.
    """
    torch = _torch()

    x = scores_or_emb
    b = x.shape[0]
    device = x.device
    labels = labels.long()

    if x.dim() == 2 and x.shape[0] == x.shape[1]:
        sim = x
    else:
        sim = (x @ x.t()).clamp(-1.0, 1.0)               # (B, B) cosine

    mask_self = torch.eye(b, dtype=torch.bool, device=device)
    rel = (labels[:, None] == labels[None, :]) & ~mask_self   # positives

    # Pairwise score differences per query: D[q, j, i] = S[q, i] - S[q, j].
    # We compute, for each query q and each candidate i, the smoothed rank
    # among (a) all others and (b) the positives only.
    aps = []
    for q in range(b):
        pos_idx = rel[q].nonzero(as_tuple=False).squeeze(1)
        if pos_idx.numel() == 0:
            continue
        s = sim[q]                                       # (B,)
        # D_i_j = s[i] - s[j]; G = sigmoid(D / tau).
        diff = (s[:, None] - s[None, :]) / float(tau)    # (B, B)
        g = torch.sigmoid(diff)
        # Zero the self term (i == j) so a candidate doesn't rank against itself.
        g = g * (~mask_self).to(g.dtype)
        # Rank among all others: 1 + sum_j G(s_j - s_i) -> we need s_i - s_j,
        # so use column i of g: g[j, i] = sigmoid((s[j]-s[i])/tau). Sum over j.
        rank_all = 1.0 + g.sum(dim=0)                    # (B,) over all j
        # Rank among positives only.
        g_pos = g[pos_idx, :]                            # (|P|, B)
        rank_pos = 1.0 + g_pos.sum(dim=0)                # (B,)
        ap = (rank_pos[pos_idx] / rank_all[pos_idx]).mean()
        aps.append(ap)

    if not aps:
        return sim.sum() * 0.0
    ap_mean = torch.stack(aps).mean()
    return 1.0 - ap_mean


# ---------------------------------------------------------------------------
# Combined recipe
# ---------------------------------------------------------------------------
def CrossModalRetrievalLoss(
    in_dim: int,
    n_classes: int,
    *,
    w_infonce: float = 1.5,
    w_arcface: float = 1.0,
    w_triplet: float = 0.5,
    temperature: float = 0.07,
    arcface_K: int = 3,
    arcface_s: float = 64.0,
    arcface_m: float = 0.5,
    triplet_margin: float = 0.1,
    cross_modal_triplet: bool = True,
):
    r"""Construct the combined cross-modal retrieval loss ``nn.Module`` (factory).

    Implements the research-recommended trio (``05_training_losses.md`` §13):

    .. math::

        L = w_{\text{nce}}\, L_{\text{sym-InfoNCE}}
          + w_{\text{arc}}\, L_{\text{SubCenterArcFace}}
          + w_{\text{tri}}\, L_{\text{batch-hard-triplet}}

    with default weights ``1.5 / 1.0 / 0.5``.  ``forward`` returns both the total
    loss and a dict of the (unweighted) components for logging.

    The InfoNCE term needs **paired** cross-modal embeddings, so ``forward``
    accepts an optional ``location_ids`` array used to build positive pairs
    (rows that share a location but differ in modality).  If no pairs can be
    formed (e.g. labels-only data), the InfoNCE term is skipped (contributes 0)
    and a warning flag is set in the components dict.

    Parameters
    ----------
    in_dim:
        Embedding dimensionality ``D``.
    n_classes:
        Number of semantic classes (for the ArcFace head).
    w_infonce, w_arcface, w_triplet:
        Loss weights (defaults 1.5 / 1.0 / 0.5).
    temperature:
        InfoNCE temperature.
    arcface_K, arcface_s, arcface_m:
        Sub-center ArcFace hyperparameters.
    triplet_margin:
        Batch-hard triplet margin.
    cross_modal_triplet:
        Prefer cross-modal triplets when modalities are supplied.

    Returns
    -------
    torch.nn.Module
        Module whose
        ``forward(embeddings, labels, modalities=None, location_ids=None)
        -> (total_loss, components_dict)``.
    """
    torch = _torch()
    import torch.nn as nn

    arcface_mod = SubCenterArcFace(
        in_dim, n_classes, K=arcface_K, s=arcface_s, m=arcface_m
    )

    class _CrossModalRetrievalLoss(nn.Module):
        """See :func:`CrossModalRetrievalLoss`."""

        def __init__(self) -> None:
            super().__init__()
            self.arcface = arcface_mod
            self.w_infonce = float(w_infonce)
            self.w_arcface = float(w_arcface)
            self.w_triplet = float(w_triplet)
            self.temperature = float(temperature)
            self.triplet_margin = float(triplet_margin)
            self.cross_modal_triplet = bool(cross_modal_triplet)

        @staticmethod
        def _build_pairs(modalities, location_ids):
            """Return (idx_a, idx_b) index tensors of cross-modal positive pairs.

            A pair shares a ``location_id`` but has different modalities. At most
            one partner is matched per anchor (first available), giving disjoint
            paired rows suitable for the symmetric-InfoNCE diagonal.
            """
            if modalities is None or location_ids is None:
                return None
            loc = [str(x) for x in location_ids]
            mod = (
                modalities.tolist()
                if hasattr(modalities, "tolist")
                else list(modalities)
            )
            buckets: dict[str, list[int]] = {}
            for i, lid in enumerate(loc):
                buckets.setdefault(lid, []).append(i)
            idx_a: list[int] = []
            idx_b: list[int] = []
            used: set[int] = set()
            for members in buckets.values():
                # Greedily pair members of differing modality within a location.
                for ii in range(len(members)):
                    a = members[ii]
                    if a in used:
                        continue
                    for jj in range(ii + 1, len(members)):
                        bb = members[jj]
                        if bb in used:
                            continue
                        if mod[a] != mod[bb]:
                            idx_a.append(a)
                            idx_b.append(bb)
                            used.add(a)
                            used.add(bb)
                            break
            if not idx_a:
                return None
            return idx_a, idx_b

        def forward(
            self,
            embeddings,
            labels,
            modalities=None,
            location_ids=None,
        ):
            components: dict[str, float] = {}
            total = embeddings.sum() * 0.0  # zero scalar on the right device

            # --- Sub-center ArcFace (always available with labels) ---
            arc = self.arcface(embeddings, labels)
            total = total + self.w_arcface * arc
            components["arcface"] = float(arc.detach())

            # --- Batch-hard triplet ---
            tri = batch_hard_triplet(
                embeddings,
                labels,
                modalities=modalities,
                margin=self.triplet_margin,
                cross_modal=self.cross_modal_triplet,
            )
            total = total + self.w_triplet * tri
            components["triplet"] = float(tri.detach())

            # --- Symmetric InfoNCE on cross-modal positive pairs ---
            pairs = self._build_pairs(modalities, location_ids)
            if pairs is not None:
                idx_a, idx_b = pairs
                ia = torch.as_tensor(idx_a, device=embeddings.device)
                ib = torch.as_tensor(idx_b, device=embeddings.device)
                nce = symmetric_infonce(
                    embeddings[ia], embeddings[ib], temperature=self.temperature
                )
                total = total + self.w_infonce * nce
                components["infonce"] = float(nce.detach())
            else:
                components["infonce"] = 0.0
                components["infonce_skipped"] = 1.0

            components["total"] = float(total.detach())
            return total, components

    return _CrossModalRetrievalLoss()
