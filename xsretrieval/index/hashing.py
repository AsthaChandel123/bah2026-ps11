"""Binary hashing: ITQ codes + Hamming search (the near-O(1) coarse filter).

Compact binary codes are the most "O(1)-flavoured" retrieval primitive (research
note ``04_fast_retrieval.md`` §7): a ``b``-bit code occupies ``b/8`` bytes and
Hamming distance is one ``XOR`` + ``popcount``.  This module provides:

* :class:`ITQHasher`  — Iterative Quantization (Gong & Lazebnik): PCA to ``b``
  dims, then a learned rotation minimising quantisation error to the
  ``{-1,+1}^b`` hypercube.  Turns existing aligned float embeddings into strong
  binary codes **without retraining a deep hashing head**.
* :class:`BinaryIndex` — builds a code database and searches it by Hamming
  distance (numpy ``popcount``, or faiss ``IndexBinaryFlat`` /
  ``IndexBinaryMultiHash`` when available — the latter is the genuine
  Multi-Index-Hashing near-O(1) lookup of Norouzi et al.).
* :func:`hamming_topk`, :func:`pack_bits`, :func:`unpack_bits` helpers.

Recommended use (research §16b): hash → fetch a few hundred candidates by Hamming
distance (near-constant time) → re-rank that pool with exact float cosine
(:func:`xsretrieval.index.rerank.exact_refine`).  You get hash-table speed *and*
float-level F1.

Multi-Index Hashing rationale
-----------------------------
Splitting each ``b``-bit code into ``m`` disjoint substrings and building ``m``
hash tables guarantees that any true neighbour within Hamming radius ``R`` matches
*exactly* in at least one substring within radius ``R/m``; querying each table and
unioning candidates yields **exact** k-NN in Hamming space with provably
sublinear (near-constant for short codes / small ``R``) work.  The portable numpy
path below implements an *exact* linear popcount scan — correct and, for the
moderate gallery sizes here, already very fast; faiss ``IndexBinaryMultiHash`` is
used transparently when present for the true MIH speed-up.

faiss is lazy/optional; the numpy path is fully functional on its own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "ITQHasher",
    "BinaryIndex",
    "hamming_topk",
    "pack_bits",
    "unpack_bits",
]

# Lookup table mapping each byte value to its set-bit count (popcount).
_POPCOUNT_TABLE = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint16)


def _faiss():
    """Try to import faiss; return the module or ``None``."""
    try:
        import faiss  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return faiss


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack a ``(N, b)`` array of ``{0,1}`` (or boolean) bits into ``(N, b/8)`` uint8.

    Bits are packed MSB-first within each byte (``np.packbits`` convention); the
    last byte is zero-padded if ``b`` is not a multiple of 8.
    """
    bits = np.asarray(bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    binary = (bits > 0).astype(np.uint8)
    return np.packbits(binary, axis=1)


def unpack_bits(packed: np.ndarray, n_bits: Optional[int] = None) -> np.ndarray:
    """Unpack ``(N, b/8)`` uint8 codes back to a ``(N, b)`` ``{0,1}`` array.

    Parameters
    ----------
    packed:
        Packed uint8 codes.
    n_bits:
        Optional original bit count to trim padding introduced by
        :func:`pack_bits`.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed[None, :]
    bits = np.unpackbits(packed, axis=1)
    if n_bits is not None:
        bits = bits[:, :n_bits]
    return bits


def _hamming_distances(query_codes: np.ndarray, db_codes: np.ndarray) -> np.ndarray:
    """Hamming distance matrix ``(Nq, N)`` between packed uint8 code sets.

    Computes ``popcount(q XOR d)`` via a 256-entry byte lookup table, fully
    vectorised over the byte dimension.  Complexity ``O(Nq * N * b/8)`` with a
    tiny constant (table gather + sum).
    """
    q = np.atleast_2d(np.asarray(query_codes, dtype=np.uint8))
    d = np.atleast_2d(np.asarray(db_codes, dtype=np.uint8))
    if q.shape[1] != d.shape[1]:
        raise ValueError(
            f"code byte-length mismatch: {q.shape[1]} != {d.shape[1]}"
        )
    # XOR each query against every db code: (Nq, N, B) — chunk over queries to
    # bound memory for large galleries.
    nq = q.shape[0]
    out = np.empty((nq, d.shape[0]), dtype=np.uint16)
    # Process in row blocks to keep the (block, N, B) tensor modest.
    block = max(1, int(2_000_000 // max(d.shape[0] * d.shape[1], 1)))
    for start in range(0, nq, block):
        qb = q[start:start + block]                       # (bq, B)
        xor = np.bitwise_xor(qb[:, None, :], d[None, :, :])  # (bq, N, B)
        out[start:start + block] = _POPCOUNT_TABLE[xor].sum(axis=2)
    return out


def hamming_topk(
    query_codes: np.ndarray,
    db_codes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-``k`` nearest neighbours in Hamming space (numpy popcount).

    Parameters
    ----------
    query_codes:
        ``(Nq, B)`` (or ``(B,)``) packed uint8 query codes.
    db_codes:
        ``(N, B)`` packed uint8 database codes.
    k:
        Neighbours per query.

    Returns
    -------
    (distances, indices):
        ``(Nq, k)`` uint16 Hamming distances and ``(Nq, k)`` int64 db indices,
        ascending by distance.
    """
    dists = _hamming_distances(query_codes, db_codes)     # (Nq, N)
    nq, n = dists.shape
    kk = min(int(k), n)
    out_idx = np.full((nq, k), -1, dtype=np.int64)
    out_dist = np.full((nq, k), np.iinfo(np.uint16).max, dtype=np.uint16)
    for i in range(nq):
        part = np.argpartition(dists[i], kk - 1)[:kk]
        order = np.argsort(dists[i][part], kind="stable")
        sel = part[order]
        out_idx[i, :kk] = sel
        out_dist[i, :kk] = dists[i][sel]
    return out_dist, out_idx


class ITQHasher:
    """Iterative Quantization hasher: float embeddings → compact binary codes.

    ITQ (Gong & Lazebnik, CVPR 2011) first PCA-projects the (centred) data to
    ``n_bits`` dimensions, then finds an orthogonal rotation ``R`` that minimises
    the quantisation error ``|| sgn(V R) - V R ||^2`` to the binary hypercube via
    alternating minimisation (an orthogonal **Procrustes** problem):

    1. ``B = sgn(Z R)``           (fix R, optimise codes)
    2. ``R = argmin ||B - Z R||`` solved by SVD of ``Z^T B`` (fix codes, rotate)

    Codes are ``sign(Z R) > 0`` packed into bits.  Far stronger than raw LSH or
    PCA-then-threshold at equal bit length, and costs only a PCA + a few rotation
    iterations (seconds–minutes), as recommended in research §9.

    Parameters
    ----------
    n_bits:
        Code length in bits (default 64). Must be ``<= input dim``.
    n_iters:
        ITQ rotation iterations (default 50; converges quickly).
    seed:
        RNG seed for the rotation initialisation (deterministic).
    """

    def __init__(self, n_bits: int = 64, n_iters: int = 50, seed: int = 0) -> None:
        self.n_bits = int(n_bits)
        self.n_iters = int(n_iters)
        self.seed = int(seed)
        self.mean_: Optional[np.ndarray] = None
        self.pca_: Optional[np.ndarray] = None     # (d, n_bits) projection
        self.rotation_: Optional[np.ndarray] = None  # (n_bits, n_bits)
        self.fitted_: bool = False

    def fit(self, emb: np.ndarray) -> "ITQHasher":
        """Fit PCA + ITQ rotation on ``emb`` (n, d)."""
        x = np.asarray(emb, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"expected (n, d) embeddings, got {x.shape}")
        n, d = x.shape
        if self.n_bits > d:
            raise ValueError(
                f"n_bits={self.n_bits} cannot exceed input dim {d}"
            )
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_

        # PCA via SVD of the centred data: top n_bits right-singular vectors.
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        pca = vt[: self.n_bits].T                          # (d, n_bits)
        v = xc @ pca                                       # (n, n_bits)

        # ITQ: alternating minimisation of quantisation loss.
        rng = np.random.default_rng(self.seed)
        r = rng.standard_normal((self.n_bits, self.n_bits))
        # Orthonormalise the initial rotation.
        u_, _, vt_ = np.linalg.svd(r)
        r = u_ @ vt_
        for _ in range(self.n_iters):
            z = v @ r
            b = np.where(z >= 0, 1.0, -1.0)                # (n, n_bits)
            # Procrustes: R = S V^T where U S V^T = SVD(B^T V).
            u2, _, vt2 = np.linalg.svd(b.T @ v)
            r = (vt2.T @ u2.T)
        self.pca_ = pca.astype(np.float32)
        self.rotation_ = r.astype(np.float32)
        self.fitted_ = True
        return self

    def project(self, emb: np.ndarray) -> np.ndarray:
        """Return the real-valued rotated projection ``(n, n_bits)`` (pre-sign)."""
        if not self.fitted_:
            raise RuntimeError("ITQHasher.project before fit()")
        x = np.asarray(emb, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z = (x - self.mean_) @ self.pca_ @ self.rotation_
        return z[0] if single else z

    def transform(self, emb: np.ndarray) -> np.ndarray:
        """Hash ``emb`` to packed ``np.uint8`` binary codes ``(n, n_bits/8)``."""
        z = self.project(emb)
        single = z.ndim == 1
        if single:
            z = z[None, :]
        bits = (z >= 0).astype(np.uint8)
        codes = pack_bits(bits)
        return codes[0] if single else codes

    def fit_transform(self, emb: np.ndarray) -> np.ndarray:
        """Fit then return packed binary codes for ``emb``."""
        return self.fit(emb).transform(emb)

    def save(self, path: str) -> None:
        """Serialise the fitted hasher to a ``.npz`` file."""
        if not self.fitted_:
            raise RuntimeError("cannot save an unfitted ITQHasher")
        np.savez(
            path,
            mean=self.mean_,
            pca=self.pca_,
            rotation=self.rotation_,
            config=np.array([self.n_bits, self.n_iters, self.seed], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "ITQHasher":
        """Load a fitted :class:`ITQHasher` from a ``.npz`` file."""
        data = np.load(path, allow_pickle=True)
        cfg = data["config"]
        obj = cls(n_bits=int(cfg[0]), n_iters=int(cfg[1]), seed=int(cfg[2]))
        obj.mean_ = np.asarray(data["mean"], dtype=np.float32)
        obj.pca_ = np.asarray(data["pca"], dtype=np.float32)
        obj.rotation_ = np.asarray(data["rotation"], dtype=np.float32)
        obj.fitted_ = True
        return obj


class BinaryIndex:
    """Hamming-distance index over packed binary codes (numpy or faiss).

    Build with packed uint8 codes (e.g. from :class:`ITQHasher`), then
    :meth:`search` for the ``k`` nearest in Hamming distance.  When faiss is
    available the index is backed by ``IndexBinaryMultiHash`` (true Multi-Index
    Hashing, near-O(1)) with a graceful fall-back to ``IndexBinaryFlat``; without
    faiss it uses the exact numpy popcount scan (:func:`hamming_topk`).

    Parameters
    ----------
    n_bits:
        Code length in bits.
    use_faiss:
        Whether to use a faiss binary index when available (default True).
    nhash:
        Number of MIH substrings / hash tables (default 4). The code is split
        into ``nhash`` substrings of ``n_bits/nhash`` bits each.
    """

    def __init__(self, n_bits: int, use_faiss: bool = True, nhash: int = 4) -> None:
        self.n_bits = int(n_bits)
        self.n_bytes = (self.n_bits + 7) // 8
        self.use_faiss_pref = bool(use_faiss)
        self.nhash = int(nhash)
        self.codes_: Optional[np.ndarray] = None
        self._faiss_index = None
        self._use_faiss = False

    def build(self, packed_codes: np.ndarray) -> "BinaryIndex":
        """Add the gallery's packed codes to the index.

        Parameters
        ----------
        packed_codes:
            ``(N, n_bits/8)`` uint8 packed codes.

        Returns
        -------
        self
        """
        codes = np.atleast_2d(np.asarray(packed_codes, dtype=np.uint8))
        if codes.shape[1] != self.n_bytes:
            raise ValueError(
                f"expected {self.n_bytes} bytes/code, got {codes.shape[1]}"
            )
        self.codes_ = np.ascontiguousarray(codes)

        if self.use_faiss_pref:
            faiss = _faiss()
            if faiss is not None:
                self._build_faiss_binary(faiss, self.codes_)
        return self

    def _build_faiss_binary(self, faiss, codes: np.ndarray) -> None:
        """Build a faiss binary index (MultiHash preferred, else Flat)."""
        try:
            if hasattr(faiss, "IndexBinaryMultiHash") and self.n_bits % self.nhash == 0:
                bits_per_hash = self.n_bits // self.nhash
                index = faiss.IndexBinaryMultiHash(
                    self.n_bits, self.nhash, bits_per_hash
                )
                index.add(codes)
                # nflip controls the search radius per substring; a small value
                # keeps it near-O(1). Set defensively if the attribute exists.
                if hasattr(index, "nflip"):
                    index.nflip = 2
            else:
                index = faiss.IndexBinaryFlat(self.n_bits)
                index.add(codes)
            self._faiss_index = index
            self._use_faiss = True
        except Exception:
            self._faiss_index = None
            self._use_faiss = False

    def search(
        self, query_codes: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` Hamming neighbours of each query code.

        Parameters
        ----------
        query_codes:
            ``(Nq, n_bits/8)`` (or ``(n_bits/8,)``) packed uint8 query codes.
        k:
            Neighbours per query.

        Returns
        -------
        (distances, indices):
            ``(Nq, k)`` Hamming distances and ``(Nq, k)`` int64 db indices,
            ascending by distance (``-1`` index / max-distance padding).
        """
        if self.codes_ is None:
            raise RuntimeError("BinaryIndex.search before build()")
        q = np.atleast_2d(np.asarray(query_codes, dtype=np.uint8))

        if self._use_faiss and self._faiss_index is not None:
            try:
                d, idx = self._faiss_index.search(np.ascontiguousarray(q), int(k))
                d = np.asarray(d, dtype=np.uint16)
                idx = np.asarray(idx, dtype=np.int64)
                # MultiHash may return -1 padding when fewer than k found; if a
                # query under-fills, transparently complete it with an exact
                # numpy scan so results are always full and correct.
                if (idx < 0).any():
                    d2, idx2 = hamming_topk(q, self.codes_, int(k))
                    fill = idx < 0
                    rows = np.where(fill.any(axis=1))[0]
                    for r in rows:
                        d[r], idx[r] = d2[r], idx2[r]
                return d, idx
            except Exception:
                pass  # fall back to numpy
        return hamming_topk(q, self.codes_, int(k))

    @property
    def uses_faiss(self) -> bool:
        """Whether a faiss binary backend is active."""
        return self._use_faiss

    def __len__(self) -> int:
        return 0 if self.codes_ is None else int(self.codes_.shape[0])
