"""Modality-balanced P x K batch sampler for cross-modal metric learning.

Pair/triplet/SupCon/Multi-Similarity/AP losses all require that each mini-batch
contains several samples per class (so in-batch positives and negatives exist).
For *cross-modal* retrieval we additionally need each class to appear in **more
than one modality** within the batch, so the loss sees genuine cross-modal
positive pairs and cross-modal hard negatives. This module implements that
"modality-balanced P x K" sampler from ``research/05_training_losses.md`` (§11.1).

The sampler is a plain Python/numpy iterator that yields **lists of dataset
indices** -- directly compatible with ``torch.utils.data.DataLoader(...,
batch_sampler=PKModalitySampler(...))`` -- and has **no torch dependency** at
import time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Optional, Sequence

import numpy as np

from .modalities import Modality, Sample

__all__ = ["PKModalitySampler"]


class PKModalitySampler:
    """Yield P-classes x K-instances batches with K split across modalities.

    Each yielded batch contains ``P`` distinct classes and, for every class,
    ``K`` instances drawn so that the available modalities are represented as
    evenly as possible. This guarantees:

    * **in-batch same-class positives** (needed by SupCon / triplet / MS / AP),
    * **in-batch cross-modal positives** (same class, different modality), and
    * **cross-modal hard negatives** (different class, any modality),

    which together make the batch a valid training signal for every loss in the
    recommended recipe.

    Two construction styles:

    * pass a ``list[Sample]`` (labels/modalities read from the samples), or
    * pass parallel ``labels`` and ``modalities`` arrays/sequences.

    The sampler exposes ``__iter__`` (yields ``list[int]`` index batches) and
    ``__len__`` (number of batches per epoch), the interface a PyTorch
    ``DataLoader`` expects for ``batch_sampler``.

    Parameters
    ----------
    samples:
        Optional ``list[Sample]``; if given, ``labels``/``modalities`` are
        derived from it.
    labels:
        Per-item integer class labels (required if ``samples`` is ``None``).
    modalities:
        Per-item modality (``Modality`` or its string value), aligned with
        ``labels``. If ``None`` (and no ``samples``), all items are treated as a
        single modality (degrades to a standard P x K sampler).
    p:
        Number of classes per batch.
    k:
        Number of instances per class per batch (batch size = ``p * k``).
    num_batches:
        Batches per epoch. If ``None``, defaults to ``ceil(num_eligible_classes /
        p)`` so an epoch roughly covers all classes once.
    seed:
        Base RNG seed (the per-epoch stream is derived from it, so iterating the
        sampler twice with the same epoch index is reproducible).
    drop_classes_with_few:
        Skip classes that have fewer than ``min_class_size`` samples (they cannot
        supply ``k`` distinct-enough instances). Such classes are excluded from
        sampling rather than silently under-filled.
    min_class_size:
        Minimum number of samples a class must have to be eligible.
    require_multimodal:
        If ``True``, only classes present in **>= 2 modalities** are eligible,
        enforcing that every batch class can yield cross-modal positives. Falls
        back to all eligible classes if fewer than ``p`` are multi-modal.
    sample_with_replacement:
        Within a class, allow drawing the same item more than once when the class
        has fewer than ``k`` items (keeps batches exactly ``p * k`` in size).
    """

    def __init__(
        self,
        samples: Optional[Sequence[Sample]] = None,
        *,
        labels: Optional[Sequence[int] | np.ndarray] = None,
        modalities: Optional[Sequence[Modality] | np.ndarray] = None,
        p: int = 16,
        k: int = 4,
        num_batches: Optional[int] = None,
        seed: int = 0,
        drop_classes_with_few: bool = True,
        min_class_size: int = 1,
        require_multimodal: bool = False,
        sample_with_replacement: bool = True,
    ) -> None:
        if samples is not None:
            labels = np.asarray([s.label for s in samples], dtype=np.int64)
            modalities = [s.modality for s in samples]
        if labels is None:
            raise ValueError(
                "PKModalitySampler needs either `samples` or `labels`."
            )
        self.labels = np.asarray(labels, dtype=np.int64)
        n = len(self.labels)

        if modalities is None:
            self.modalities = np.zeros(n, dtype=object)
        else:
            self.modalities = np.asarray(
                [Modality(m).value for m in modalities], dtype=object
            )
        if len(self.modalities) != n:
            raise ValueError("labels and modalities must have the same length")

        if p < 1 or k < 1:
            raise ValueError("p and k must be >= 1")
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.sample_with_replacement = sample_with_replacement
        self.min_class_size = int(min_class_size)
        self.require_multimodal = require_multimodal

        # Build per-class -> per-modality index lists.
        # class_modal_index[label][modality_value] = ndarray of dataset indices.
        self._class_modal: dict[int, dict[object, np.ndarray]] = {}
        tmp: dict[int, dict[object, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx in range(n):
            tmp[int(self.labels[idx])][self.modalities[idx]].append(idx)
        for label, modal_map in tmp.items():
            self._class_modal[label] = {
                mod: np.asarray(ids, dtype=np.int64)
                for mod, ids in modal_map.items()
            }

        # Determine eligible classes.
        self._eligible = self._compute_eligible(drop_classes_with_few)
        if len(self._eligible) == 0:
            raise ValueError(
                "no eligible classes for sampling (check labels / min_class_size)"
            )

        if num_batches is None:
            num_batches = max(
                1, int(np.ceil(len(self._eligible) / self.p))
            )
        self.num_batches = int(num_batches)
        self._epoch = 0

    # -- eligibility ------------------------------------------------------
    def _class_size(self, label: int) -> int:
        return int(sum(len(v) for v in self._class_modal[label].values()))

    def _num_modalities(self, label: int) -> int:
        return len(self._class_modal[label])

    def _compute_eligible(self, drop_few: bool) -> list[int]:
        """Return the list of class labels usable for sampling."""
        eligible: list[int] = []
        for label in self._class_modal:
            if drop_few and self._class_size(label) < self.min_class_size:
                continue
            eligible.append(label)
        eligible.sort()

        if self.require_multimodal:
            multimodal = [
                lbl for lbl in eligible if self._num_modalities(lbl) >= 2
            ]
            # Only enforce if we still have enough classes to form a batch.
            if len(multimodal) >= min(self.p, 1):
                return multimodal
        return eligible

    # -- per-class, modality-balanced draw --------------------------------
    @staticmethod
    def _modality_quota(k: int, n_mod: int) -> list[int]:
        """Split ``k`` instances across ``n_mod`` modalities as evenly as possible.

        e.g. ``k=4, n_mod=3 -> [2, 1, 1]`` (the extra goes to the first
        modalities). This is what guarantees multiple modalities per class in the
        batch.
        """
        base = k // n_mod
        rem = k % n_mod
        return [base + (1 if i < rem else 0) for i in range(n_mod)]

    def _draw_class(
        self, label: int, rng: np.random.Generator
    ) -> list[int]:
        """Draw ``k`` indices for ``label``, balanced across its modalities."""
        modal_map = self._class_modal[label]
        mods = list(modal_map.keys())
        rng.shuffle(mods)
        quota = self._modality_quota(self.k, len(mods))

        chosen: list[int] = []
        deficit = 0
        for mod, want in zip(mods, quota):
            want += deficit
            deficit = 0
            pool = modal_map[mod]
            if want <= 0:
                continue
            if len(pool) >= want:
                pick = rng.choice(pool, size=want, replace=False)
            elif self.sample_with_replacement:
                pick = rng.choice(pool, size=want, replace=True)
            else:
                pick = pool.copy()
                deficit = want - len(pool)  # carry shortfall to next modality
            chosen.extend(int(x) for x in pick)

        # If a shortfall remains (e.g. last modality too small), top up from the
        # whole class pool so the batch stays exactly p*k.
        if len(chosen) < self.k:
            all_idx = np.concatenate(list(modal_map.values()))
            need = self.k - len(chosen)
            replace = self.sample_with_replacement or len(all_idx) < need
            extra = rng.choice(all_idx, size=need, replace=replace)
            chosen.extend(int(x) for x in extra)
        return chosen[: self.k]

    # -- iteration --------------------------------------------------------
    def _make_batch(self, rng: np.random.Generator) -> list[int]:
        """Sample one P x K batch of dataset indices."""
        n_classes = min(self.p, len(self._eligible))
        classes = rng.choice(self._eligible, size=n_classes, replace=False)
        batch: list[int] = []
        for label in classes:
            batch.extend(self._draw_class(int(label), rng))
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        # Derive a per-epoch generator so successive epochs differ but each is
        # reproducible from (seed, epoch).
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.num_batches):
            yield self._make_batch(rng)

    def __len__(self) -> int:
        return self.num_batches

    @property
    def batch_size(self) -> int:
        """Nominal batch size (``p * k``)."""
        return self.p * self.k

    @property
    def eligible_classes(self) -> list[int]:
        """The class labels the sampler draws from."""
        return list(self._eligible)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index (for reproducible, epoch-varying shuffling).

        Mirrors ``DistributedSampler.set_epoch``; call before each epoch so the
        sampled batches vary deterministically with ``epoch``.
        """
        self._epoch = int(epoch)
