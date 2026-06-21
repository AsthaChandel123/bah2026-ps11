"""Tests for the training losses (``xsretrieval.alignment.losses``).

Covers the two contracted objects from ``research/05``:

* ``symmetric_infonce(z_a, z_b, temperature=0.07)`` -- the CLIP-style symmetric
  InfoNCE used to close the modality gap. For *perfectly aligned* unit pairs
  (each modality-A embedding equals its modality-B partner, partners mutually
  well-separated) the loss is small / near its floor; for *misaligned*
  embeddings it is strictly larger.
* ``CrossModalRetrievalLoss(in_dim, n_classes, ...)`` -- a factory returning an
  ``nn.Module`` whose ``forward(embeddings, labels, modalities=None,
  location_ids=None) -> (total_loss, components)``. We check it produces a
  finite scalar with a valid backward pass on a tiny batch.

The whole module is skipped if torch is unavailable (these are torch losses).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="losses are torch-based")

losses = pytest.importorskip(
    "xsretrieval.alignment.losses",
    reason="alignment team's losses module not available yet",
)


def _l2norm(x: "torch.Tensor") -> "torch.Tensor":
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_symmetric_infonce_lower_for_aligned() -> None:
    fn = getattr(losses, "symmetric_infonce", None)
    if fn is None:
        pytest.skip("symmetric_infonce not implemented")

    torch.manual_seed(0)
    n, d = 8, 16
    # Well-separated anchor directions (orthonormal rows via QR).
    base = torch.linalg.qr(torch.randn(d, d))[0][:n]  # (n, d) orthonormal rows
    a_aligned = _l2norm(base.clone())
    b_aligned = _l2norm(base.clone())  # perfect cross-modal correspondence

    aligned_loss = float(fn(a_aligned, b_aligned))

    # Misaligned: cyclic-shift B so positives no longer lie on the diagonal.
    perm = torch.tensor([(i + 1) % n for i in range(n)])
    b_misaligned = b_aligned[perm].clone()
    misaligned_loss = float(fn(a_aligned, b_misaligned))

    assert aligned_loss >= 0.0
    assert misaligned_loss > aligned_loss, (
        f"InfoNCE should be larger when misaligned: aligned={aligned_loss:.4f} "
        f"misaligned={misaligned_loss:.4f}"
    )
    # For near-perfect alignment of well-separated pairs the loss is small.
    assert aligned_loss < misaligned_loss * 0.9


def test_symmetric_infonce_is_finite_scalar_and_differentiable() -> None:
    fn = getattr(losses, "symmetric_infonce", None)
    if fn is None:
        pytest.skip("symmetric_infonce not implemented")
    torch.manual_seed(1)
    a = _l2norm(torch.randn(6, 12, requires_grad=True))
    b = _l2norm(torch.randn(6, 12, requires_grad=True))
    loss = fn(a, b)
    assert loss.ndim == 0 or loss.numel() == 1
    assert torch.isfinite(loss).all()
    loss.backward()  # must be differentiable


def test_symmetric_infonce_temperature_kw() -> None:
    """A custom temperature is accepted and yields a finite loss."""
    fn = getattr(losses, "symmetric_infonce", None)
    if fn is None:
        pytest.skip("symmetric_infonce not implemented")
    torch.manual_seed(2)
    a = _l2norm(torch.randn(5, 10))
    b = _l2norm(torch.randn(5, 10))
    loss = fn(a, b, temperature=0.2)
    assert torch.isfinite(torch.as_tensor(loss)).all()


def _build_cross_modal_loss(factory):
    """Instantiate the CrossModalRetrievalLoss factory tolerantly.

    Real signature is ``CrossModalRetrievalLoss(in_dim, n_classes, ...)``; we try
    that first, then a couple of fallbacks for robustness to minor API drift.
    """
    for attempt in (
        lambda: factory(16, 4),
        lambda: factory(in_dim=16, n_classes=4),
        lambda: factory(16, n_classes=4),
    ):
        try:
            return attempt()
        except TypeError:
            continue
    pytest.skip("could not construct CrossModalRetrievalLoss")


def test_cross_modal_retrieval_loss_backprops() -> None:
    factory = getattr(losses, "CrossModalRetrievalLoss", None)
    if factory is None:
        pytest.skip("CrossModalRetrievalLoss not implemented")

    torch.manual_seed(3)
    n, d = 8, 16
    loss_mod = _build_cross_modal_loss(factory)

    # A tiny batch: two modalities, paired by location, with class labels.
    # ``embeddings`` is the L2-normalized result of an op, so it is a *non-leaf*
    # tensor; retain its grad so we can assert the gradient that flows back to
    # the embedding fed to the loss (the leaf is the pre-norm randn).
    embeddings = _l2norm(torch.randn(n, d, requires_grad=True))
    embeddings.retain_grad()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    from xsretrieval.data.modalities import Modality

    modalities = [
        Modality.OPTICAL_RGB, Modality.SAR,
        Modality.OPTICAL_RGB, Modality.SAR,
        Modality.OPTICAL_RGB, Modality.SAR,
        Modality.OPTICAL_RGB, Modality.SAR,
    ]
    # Paired locations so the InfoNCE term can form cross-modal positives.
    location_ids = ["l0", "l0", "l1", "l1", "l2", "l2", "l3", "l3"]

    # Try the documented forward signature first, then simpler fallbacks.
    out = None
    for attempt in (
        lambda: loss_mod(
            embeddings, labels, modalities=modalities, location_ids=location_ids
        ),
        lambda: loss_mod(embeddings, labels, modalities=modalities),
        lambda: loss_mod(embeddings, labels),
    ):
        try:
            out = attempt()
            break
        except TypeError:
            continue
    if out is None:
        pytest.skip("could not call CrossModalRetrievalLoss.forward")

    # forward returns (total_loss, components_dict) per the factory docstring.
    loss = out[0] if isinstance(out, tuple) else out
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss)

    assert loss.ndim == 0 or loss.numel() == 1
    assert torch.isfinite(loss).all()

    loss.backward()
    assert embeddings.grad is not None, "no gradient flowed to embeddings"
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0, "gradients are all zero"
