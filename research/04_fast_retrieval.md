# 04 — Fast Retrieval Techniques (Near-O(1) / Sublinear ANN) for Cross-Modal Satellite Image Retrieval

**Problem context (BAH 2026 PS-11):** Cross-modal satellite image retrieval. The scoring counts **F1@5 / F1@10 (same-modal AND cross-modal)** *and* **average retrieval time per query**. Cross-modal is weighted higher. The user wants the **fastest platform — "O(1) techniques"**.

**Core architectural premise of this doc:** A separate research thread covers the *embedding model* (the backbone + cross-modal alignment that maps optical / multispectral / SAR into ONE shared, L2-normalized embedding space). This doc assumes that space already exists and focuses **only on the search/retrieval layer**: given a query embedding, return the top-k gallery neighbors as fast as possible while keeping F1 high.

> **TL;DR recommendation (jump to [§16](#16-final-recommended-design)):** Because alignment puts every modality in **one space**, build **ONE shared FAISS index over all-modality gallery embeddings** (tag each vector with its modality id so you can filter to optical-only / SAR-only / cross-modal as the metric requires). L2-normalize → use **inner product = cosine**. For the gallery sizes in this challenge (likely ≤ ~50k–500k), the winning latency-vs-F1 design is a **hybrid: a coarse, near-O(1)-ish binary-hash / IVF filter to fetch a small candidate pool, then an exact float re-rank of that pool, then an optional cheap k-reciprocal re-rank on the top ~20**. Exact `index_factory` strings per size in [§14](#14-index-factory-cheatsheet-per-gallery-size).

---

## 0. Why retrieval speed is "won or lost" at the index layer

For a gallery of `N` vectors of dimension `d`:

- **Brute force (exact)** computes `N` distances per query → **O(N·d)**. Fine for tiny `N`, but it is linear in `N`.
- **Sublinear ANN** (IVF, graph, multi-index) touches only a fraction of `N` → effectively **O(√N·d)**, **O(log N · d)**, or a small constant number of cells.
- **Hash-table lookup (binary codes + multi-index hashing)** is the closest thing to true **O(1)**: you hash the query into buckets and read out the bucket contents; expected work is roughly constant for a fixed Hamming radius, independent of `N` (for uniformly distributed codes it is provably **sublinear** and in practice near-constant).

The single most important lever is **how many gallery vectors you actually compare against**. Everything below is a way to shrink that number while losing as little recall as possible. The second lever is **how cheap each comparison is** (float L2 vs. PQ table lookup vs. popcount Hamming).

**Two cheap, universal wins applied throughout:**
1. **L2-normalize all embeddings, then use inner product (cosine).** With unit vectors, `cosine(a,b) = a·b`, and maximizing inner product = minimizing L2. In FAISS: call `faiss.normalize_L2(x)` on gallery AND queries, then use `METRIC_INNER_PRODUCT`. Forgetting to normalize *either* side breaks the equivalence. ([FAISS MetricType wiki](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances); [Milvus note](https://milvus.io/ai-quick-reference/what-adjustments-need-to-be-made-to-an-ann-algorithm-when-switching-from-euclidean-to-cosine-similarity-consider-that-cosine-similarity-can-be-achieved-via-normalized-vectors-and-euclidean-distance))
2. **Reduce dimensionality with PCA (e.g. 512/768 → 128–256, or even 64) before indexing.** Every distance is then cheaper *and* every index is smaller. Reported case studies: 256→64 with >95% variance kept gives ≈4× capacity and <5% information loss; one production case got **16× memory reduction and 6× faster search at recall ≈0.96**. Implement as a FAISS pre-transform: `PCAR128,...` (the `R` rotates/whitens, which also helps later PQ/OPQ). Caveat: PCA too aggressively, or PCA on already-compact embeddings, can *hurt* recall — validate on your data. ([FAISS PCA discussion](https://github.com/facebookresearch/faiss/issues/400); [dim-reduction case study](https://www.sciencedirect.com/science/article/pii/S2666912922000241))

---

## 1. FAISS `IndexFlat` — exact brute force (the baseline)

**How it works:** Stores all `N` vectors uncompressed; for each query computes the exact distance to every vector and returns the true top-k. `IndexFlatIP` (inner product, for cosine after normalization) or `IndexFlatL2` (Euclidean).

- **Query complexity:** **O(N·d)** — linear. No build/train step.
- **Build cost:** ~0 (just `add`). No training.
- **Memory:** `4·d` bytes/vector (fp32). For 50k × 256-d ≈ 51 MB.
- **Recall/accuracy:** **100% (ground truth).** This is the reference every ANN method is measured against.
- **When it's the right call:** **Small galleries (≈1k–50k).** On a CPU, exact search over tens of thousands of 128–256-d vectors is sub-millisecond-to-low-ms per query *batched*, and FAISS hand-optimizes it. For PS-11, if the gallery is only a few thousand images, **`IndexFlatIP` is very likely fast enough and gives the best possible F1** — do not over-engineer. Use it as your accuracy ceiling and latency reference. FAISS guidance: for ~1k–10k searches, "use Flat." ([Guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index))
- **API:**
  ```python
  index = faiss.IndexFlatIP(d)         # cosine after faiss.normalize_L2
  index.add(gallery)                   # gallery: (N, d) float32, L2-normalized
  D, I = index.search(query, k)        # query: (nq, d) float32, L2-normalized
  ```
- **IP vs L2:** With unit-norm vectors they rank identically (`||a-b||² = 2 - 2a·b`). Prefer **IP** because most contrastive/CLIP-style alignment models are trained with cosine, and it lets you reuse the same metric for the ANN indexes below.

---

## 2. FAISS `IVF` (inverted file, coarse quantizer) — sublinear via cell pruning

**How it works:** k-means clusters the gallery into `nlist` Voronoi cells (the *coarse quantizer*). Each vector is stored in its nearest cell's inverted list. At query time, find the `nprobe` nearest centroids and scan only those lists (exact float distances inside `IVFx,Flat`). You compare against ≈ `N·nprobe/nlist` vectors instead of `N`.

- **Query complexity:** ≈ **O(nlist·d)** for centroid assignment **+ O(N·nprobe/nlist · d)** for list scan. With `nlist ≈ √N` this is the classic **O(√N·d)** sublinear regime.
- **Build cost:** k-means training on a sample (FAISS recommends 30k–256k training vectors for `nlist` up to a few thousand). Training is the main one-time cost.
- **Memory:** `IVFx,Flat` keeps full fp32 vectors: `4·d + 8` bytes/vec (8 bytes for the id). Same order as Flat.
- **Recall/accuracy:** Tunable. `nprobe=1` is fast but lossy; raising `nprobe` monotonically increases recall toward 100% (at `nprobe=nlist` it equals brute force). You trade `nlist` (speed of pruning) against `nprobe` (recall).
- **Tuning rules (well established):**
  - `nlist = C·√N` with **C ≈ 4–16** (FAISS guideline; `C≈10` is a common default). E.g. N=50k → √N≈224 → `nlist≈1024–4096`; N=500k → √N≈707 → `nlist≈4096–16384`.
  - Start `nprobe` at ~`nlist/64` … `nlist/16` and sweep upward until F1 stops improving. Typical sweet spots: `nprobe = 8–32` for `nlist=1024`, `16–64` for `nlist=4096`.
  - **You must `train` then `add`** (IVF needs centroids before adding).
- **API / factory:**
  ```python
  index = faiss.index_factory(d, "IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
  index.train(train_sample)            # representative subset of gallery
  index.add(gallery)
  index.nprobe = 16                    # the speed/recall knob
  D, I = index.search(query, k)
  ```
- **Notes:** IVF is the workhorse for "great quality, good speed, reasonable memory." It is GPU-friendly. ([Pinecone IVF](https://www.pinecone.io/learn/series/faiss/vector-indexes/); [Guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index))

---

## 3. FAISS `IVFPQ` + `OPQ` — product quantization (compressed, memory-frugal, fast)

**How it works:** Same IVF cell pruning, but inside each list the vectors are **Product-Quantized**: split each `d`-vector into `M` sub-vectors, quantize each to one of `2^nbits` (usually 256, i.e. `nbits=8`) centroids → an `M`-byte code. Distances are estimated by **Asymmetric Distance Computation (ADC)**: precompute a lookup table of query-subvector ↔ codebook-centroid distances once per query, then each candidate distance is `M` table lookups + adds (no full float math). **OPQ** (`OPQ` pre-transform) learns a rotation that makes the subspaces more independent, substantially improving PQ accuracy; it doubles as dimensionality handling (`d` should be a multiple of `M`, ideally `4M`).

- **Query complexity:** centroid assignment `O(nlist·d)` + list scan where each distance is `O(M)` table lookups. Far cheaper per-candidate than float; great for large `N`.
- **Build cost:** OPQ rotation + PQ codebook training (k-means per subquantizer). Heavier than plain IVF but one-time.
- **Memory:** **the headline feature.** `ceil(M·nbits/8) + 8` bytes/vec. With `M=32, nbits=8` → 32 bytes + id vs. `4·d` for Flat. For d=256 that's 32 B vs 1024 B = **~32× smaller.** Enables millions of vectors in RAM.
- **Recall/accuracy:** Lower than IVFFlat/HNSW because codes are lossy; recover most of it by (a) bigger `M`, (b) OPQ, (c) **re-ranking the top candidates with exact distances** (see `RFlat`/`IndexRefineFlat`, [§13](#13-re-ranking) and FastScan below).
- **Tuning:** choose `M` to hit a memory budget and so `d % M == 0` (use OPQ to set an `M·k`-dim output). `nbits=8` is standard. Same `nlist`/`nprobe` rules as IVF.
- **API / factory:**
  ```python
  # OPQ rotates to 64-d, IVF with 4096 cells, PQ into 32 sub-codes (8 bits each)
  index = faiss.index_factory(d, "OPQ32_64,IVF4096,PQ32", faiss.METRIC_INNER_PRODUCT)
  index.train(train_sample); index.add(gallery); index.nprobe = 32
  ```
- **Verdict for PS-11:** Only needed if the gallery is **large (≥ ~500k–1M)** or memory is tight. For small galleries the lossy codes cost F1 for no real latency benefit — prefer Flat/IVFFlat/HNSW. ([IVFPQ explainer](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/); [Pinecone PQ](https://www.pinecone.io/learn/series/faiss/product-quantization/))

### 3b. FAISS **FastScan** (`PQ…x4fs`, `IVF…,PQ…x4fsr`) — SIMD-accelerated PQ, *the* speed king on CPU
A 4-bit PQ variant (`nbits=4`, 16 LUT entries) that keeps lookup tables **in SIMD registers** and processes vectors in blocks (`bbs=32` default; 64/96 possible). It replaces cache accesses with in-register shuffles.

- **Speed:** **4–6× faster than ordinary ADC scan, returning the exact same results.** FAISS reports **up to 1M QPS** (no rerank), and **280k QPS at 1-recall@1 = 0.9 with reranking — 2× faster than HNSW's 140k QPS** on the same task.
- **Factory:** `PQ32x4fs` (flat), `IVF2048,PQ32x4fs`, or with residual `IVF2048,PQ32x4fsr`. Add an exact rerank stage with `IVF2048,PQ32x4fsr,RFlat` or wrap in `IndexRefineFlat`; query `k·k_factor` then refine to `k`.
- **Why it matters here:** If you want PQ's compactness *and* top-tier CPU latency, **FastScan + light refine is often the single best CPU operating point** — frequently beating HNSW on QPS-at-fixed-recall. ([FastScan wiki](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)); [ScaNN vs 4-bit PQ](https://medium.com/@kumon/similarity-search-scann-and-4-bit-pq-ab98766b32bd))

---

## 4. FAISS `HNSW` (graph ANN) — O(log N), highest recall, no training

**How it works:** Builds a multi-layer "navigable small world" proximity graph. Search greedily descends from a top entry point, hopping to ever-closer neighbors; the layered structure gives a logarithmic "zoom-in." `IndexHNSWFlat` stores full vectors at the base layer for exact final distances.

- **Query complexity:** **O(log N · d)** — because each layer's node degree is capped, total search is logarithmic in `N`. This is the canonical "fast + high recall" choice.
- **Build cost:** No k-means training, but graph construction is `O(N·log N)` and `efConstruction`-sensitive — building is slower/heavier than IVF for large `N`.
- **Memory:** `~4·d + M·2·4` bytes/vec for `IndexHNSWFlat` (full vectors + graph links); **the most memory-hungry** of the float indexes. Scaling `M` raises memory linearly. (`efSearch`/`efConstruction` cost **no** extra memory.)
- **Recall/accuracy:** Excellent — typically reaches 0.95–0.99 recall at low latency; often the best recall-at-latency for medium `N` (when memory is fine).
- **Tuning:**
  - `M` = graph degree, **8–64** (default 32). Higher M = better recall + more memory + slower build.
  - `efConstruction` (build depth, e.g. 40–200): higher = better graph, slower build, no query cost.
  - `efSearch` (query depth, e.g. 16–256): **the runtime speed/recall knob** — raise it for recall, lower it for latency. Must be ≥ k.
- **Caveat:** HNSW recall can degrade on high-intrinsic-dimensional or clustered data; validate. It also can't delete vectors and doesn't support `add_with_ids` (wrap with `IDMap` if you need custom ids).
- **API / factory:**
  ```python
  index = faiss.index_factory(d, "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
  index.hnsw.efConstruction = 80
  index.add(gallery)                   # no train needed
  index.hnsw.efSearch = 64
  D, I = index.search(query, k)
  ```
- **Memory-lean variants:** `HNSW32,PQ` / `HNSW32,SQ8` store compressed base vectors. ([Pinecone HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/); [FAISS indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes); [recall caveats](https://ranjankumar.in/hnsw-vector-search-recall-production))

---

## 5. FAISS `IVF…_HNSW` hybrid coarse quantizer — fast centroid assignment at scale

**How it works:** In plain IVF, finding the `nprobe` nearest of `nlist` centroids is itself a brute-force over `nlist` (cost `O(nlist·d)`), which becomes the bottleneck when `nlist` is large (65k–1M). Replace that linear centroid search with an **HNSW graph over the centroids**: `IVF65536_HNSW32`. Now coarse assignment is `O(log nlist · d)`.

- **Query complexity:** **O(log nlist · d)** coarse + the usual list scan. Removes the large-`nlist` bottleneck.
- **Build cost:** trains IVF centroids *and* builds an HNSW over them.
- **Memory:** IVF lists (Flat or PQ) + a small HNSW over centroids (negligible vs. the dataset).
- **Recall/accuracy:** Same as the underlying IVF/IVFPQ at a given `nprobe`; you just reach a high `nlist` affordably.
- **When:** **Large galleries (≥1M)** with many cells — this is exactly the FAISS recommendation for 1M–1B (`IVF65536_HNSW32`, `IVF262144_HNSW32`, `IVF1048576_HNSW32`). For PS-11's likely scale you won't need it, but it's the right answer if the gallery balloons.
- **Factory:** `OPQ32_64,IVF65536_HNSW32,PQ32` (billion-scale memory-frugal) or `IVF65536_HNSW32,Flat`. ([Guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index))

---

## 6. ScaNN (Google Anisotropic Vector Quantization) — SOTA recall-vs-speed on CPU

**How it works:** Tree (partitioning, like IVF) → **Anisotropic** vector quantization (AH = asymmetric hashing) → optional exact **reordering**. The key idea: a quantization loss that weights errors **along the query direction** more (errors that change the *inner product ranking* matter more than errors orthogonal to it). This estimates top-k inner products more accurately than isotropic PQ at the same code size.

- **Query complexity:** sublinear like IVF (tree partitioning) but with a **better accuracy-per-candidate**, so it needs fewer candidates for the same recall.
- **Build cost:** partition training + AH codebooks; comparable to OPQ+IVFPQ.
- **Memory:** compact AH codes (similar order to PQ).
- **Recall/accuracy:** **State-of-the-art in the high-recall regime on CPU.** On glove-100-angular (ann-benchmarks) ScaNN handled **~2× the QPS of the next-fastest library at a given accuracy**, beating 11 baselines incl. FAISS and hnswlib.
- **Important caveat for PS-11:** ScaNN is **x86/AVX-only**, TensorFlow/NumPy API, and **optimized for batch throughput, not single-query latency** — and the metric here is *avg time per query (batch=1)*. It can still win, but **benchmark it at batch=1 against FAISS** before committing; on some single-query workloads FAISS FastScan/HNSW is simpler and competitive.
- **API (tree-AH + reorder):**
  ```python
  import scann
  searcher = (scann.scann_ops_pybind.builder(gallery, 10, "dot_product")
      .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000)
      .score_ah(2, anisotropic_quantization_threshold=0.2)
      .reorder(100)            # exact rerank of top-100 → big recall boost
      .build())
  neighbors, distances = searcher.search_batched(queries)   # or .search(q) for one
  ```
  - `num_leaves` ≈ partition count (like `nlist`); `num_leaves_to_search` ≈ `nprobe` (tune to recall target); `reorder(R)` with `R > k` is **strongly recommended** when using AH. ([ScaNN paper / MLR](http://proceedings.mlr.press/v119/guo20h/guo20h.pdf); [Google blog](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/); [FAISS-vs-ScaNN study](https://arxiv.org/html/2507.16978v1))

---

## 7. Deep Semantic HASHING → binary codes + Hamming, and **Multi-Index Hashing (MIH)** for **true near-O(1)** lookup ★ (the closest to O(1) — emphasized)

This is the most "O(1)-flavored" family and directly fits "fastest platform." Two parts: **(A) learn good binary codes**, **(B) search them in near-constant time**.

### 7A. Learning compact binary hash codes for remote sensing
Train the alignment network with a **hashing head** so each image maps to a `b`-bit code (`b` ∈ {16,32,48,64,128}). Because all modalities share the embedding space, codes are **cross-modal comparable**. Methods, all directly applicable to RS:
- **DPSH** (Deep Pairwise-Supervised Hashing) — joint feature + hash-code learning by maximizing likelihood of pairwise similarities; `sign()` made differentiable by fixing binary codes during the gradient step. ([DPSH](https://arxiv.org/abs/1511.03855))
- **DHN / HashNet** — pairwise losses with continuation/`tanh→sign` annealing to fight quantization error and class imbalance.
- **GreedyHash** — point-wise loss + classification term; `sign()` integrated with a **greedy back-prop** so gradients pass through. ([survey](https://arxiv.org/pdf/2510.27232))
- **CSQ (Central Similarity Quantization, CVPR 2020)** — generates **hash centers** from a **Hadamard matrix** (maximally separated points in Hamming space) and pulls each class's codes to its center. **+3–20% mAP over prior SOTA**; very strong, simple, label-driven. Ideal when you have land-cover/land-use labels (which PS-11 allows). ([CSQ paper](https://arxiv.org/abs/1908.00347); [CSQ code](https://github.com/swuxyj/DeepHash-pytorch/blob/master/CSQ.py))
- **DSH** and metric-learning / contrastive / self-supervised deep hashing variants are well studied **specifically for remote sensing** (e.g. deep-hash RS retrieval with semantic cues; deep contrastive self-supervised hashing for RS using only unlabeled images). ([RS deep hashing](https://www.mdpi.com/2072-4292/14/24/6358); [contrastive SSL hashing for RS](https://www.mdpi.com/2072-4292/14/15/3643))

> **Practical "greedy hash" shortcut without retraining:** if you already have good float embeddings, learn codes post-hoc with **ITQ** ([§9](#9-itq--pca-hashing--learned-compact-binary-codes)) — much cheaper than retraining and surprisingly strong.

### 7B. Searching the codes — why it's near-O(1)
- **Naïve linear Hamming scan** (`IndexBinaryFlat`): `b`-bit codes use only `b/8` bytes; Hamming = XOR + **popcount** (1 CPU instruction), heavily optimized (esp. 256-bit). This is **O(N)** but with a *tiny* constant — for `b=64` you scan 8 bytes/vector. Often this alone is faster than float ANN for moderate `N`.
- **Multi-Index Hashing (MIH)** — the **true sublinear / near-O(1)** method (Norouzi et al., PAMI 2014): split each `b`-bit code into `m` disjoint substrings, build **`m` hash tables** (one per substring). For a radius-`R` Hamming search, any true neighbor must match exactly in at least one substring within radius `R/m`; query each table for its substring, union the candidates, then verify true Hamming distance on that small set. **Exact** k-NN in Hamming space.
  - **Complexity:** provably **sublinear for uniformly distributed codes**; empirically **dramatic speedups over linear scan on up to 1 BILLION codes** at 64/128/256 bits. With short codes and small `R`, each table lookup is essentially a **hash-bucket read → O(1)**.
  - **In FAISS:** `IndexBinaryMultiHash` (the MIH-style multi-hash) and `IndexBinaryHash`; also `IndexBinaryIVF` (cluster binary codes) and `IndexBinaryHNSW` (graph over binary codes). FAISS benchmark note: for small Hamming radius `IndexBinaryHash` minimizes distance computations; beyond radius ~32 `IndexBinaryIVF` wins. ([MIH paper](https://www.cs.toronto.edu/~fleet/research/Papers/MIH-pami2014.pdf); [MIH code](https://github.com/norouzi/mih); [FAISS binary indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes); [binary benchmark](https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark))
- **API:**
  ```python
  # codes: (N, b/8) uint8  (packed bits); queries likewise
  bindex = faiss.IndexBinaryMultiHash(b, nhash, bits_per_hash)  # MIH-style hash tables
  # or: bindex = faiss.IndexBinaryHash(b, bits_per_hash)
  # or: bindex = faiss.IndexBinaryFlat(b)        # exact linear popcount scan
  bindex.add(codes)
  Dham, I = bindex.search(qcodes, k)             # Dham = Hamming distances
  ```
- **Why emphasize for PS-11:** This is the design that minimizes *avg query time* the most. The recommended pattern: **hash filter (near-O(1)) → fetch a few hundred candidates → re-rank with exact float cosine** ([§16](#16-final-recommended-design)). You get hash-table speed *and* float-level F1.

---

## 8. Locality-Sensitive Hashing (LSH) families; FAISS `IndexLSH`

**How it works:** Random (or structured) projections hash so that nearby vectors collide with high probability. Classic data-independent baseline. FAISS `IndexLSH` actually uses a *better-than-vanilla* construction: **orthogonal projectors** when `nbits ≤ d`, or a **tight frame** when `nbits > d`, then binarizes; search is Hamming.

- **Query complexity:** with hash tables, bucket lookup is ~**O(1)** expected; FAISS `IndexLSH` itself does a Hamming comparison (so O(N) with tiny constant unless you put codes in MIH).
- **Build cost:** trivial (random/orthogonal projection — data-independent).
- **Memory:** `ceil(nbits/8)` bytes/vec.
- **Recall/accuracy:** **lower than learned hashing (ITQ/deep) at equal bits** because projections ignore the data distribution. Needs more bits/tables for the same recall.
- **When:** quick baseline, or when you can't train. For PS-11, **prefer ITQ or deep hashing** over raw LSH — they dominate at equal code length. ([FAISS indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes))
- **API:** `index = faiss.IndexLSH(d, nbits)` → `index.add(x)` → `index.search(q, k)`.

---

## 9. ITQ / PCA-hashing → learned compact binary codes (cheap, no deep retraining)

**How it works (Gong & Lazebnik):** PCA-project to `b` dims, then find a **rotation `R`** that minimizes quantization error to the `{-1,+1}^b` hypercube vertices (solved by alternating minimization — an orthogonal **Procrustes** problem). Output: high-quality `b`-bit codes from existing float embeddings. Can use PCA (unsupervised) or CCA (supervised) as the projection.

- **Query complexity:** produces binary codes → feed to Hamming/MIH ([§7B](#7b-searching-the-codes--why-its-near-o1)) for near-O(1) search.
- **Build cost:** PCA + a few rotation iterations — **seconds to minutes**, far cheaper than training a deep hashing net.
- **Memory:** `b/8` bytes/vec.
- **Recall/accuracy:** **significantly better than LSH and PCA-then-threshold** at the same bits; a strong, classic baseline for compact codes.
- **When for PS-11:** the **pragmatic path to binary codes if you don't want to add a deep hashing head** — take your aligned float embeddings, fit ITQ to e.g. 64 bits, search with MIH, re-rank float. ([ITQ paper](https://slazebni.cs.illinois.edu/publications/ITQ.pdf))
- **FAISS:** `faiss.ITQMatrix` / the `"ITQ64"` transform, or precede PQ with rotation; combine as `PCAR64,ITQ64,...` style pre-transforms then a binary index.

---

## 10. Inverted Multi-Index (IMI) + ADC/SDC — denser partitioning, shorter candidate lists

**How it works (Babenko & Lempitsky):** Generalizes IVF by product-quantizing the *coarse* quantizer: split the vector in two halves, each with a codebook of `T` sub-centroids → **`C = T²` cells** from only `2T` centroids. This **very fine partition** means a tiny fraction of the dataset must be scanned for a given recall, giving **shorter, higher-recall candidate lists** than IVF.
- **Distance computation inside lists:** **ADC** (asymmetric — query stays float, DB is quantized; more accurate) vs **SDC** (symmetric — both quantized; faster table-only). Prefer **ADC** for accuracy.
- **Query complexity:** sublinear; "multi-D-ADC" was SOTA on 1B SIFT, returning much shorter candidate lists at higher recall than single inverted index, with only a few % memory overhead.
- **When:** **very large** datasets. FAISS exposes it via `IMI2x<bits>` coarse quantizers, e.g. `IMI2x8,PQ32` (= 2 codebooks × 2⁸ → 65,536 cells). Overkill for PS-11 scale but the right tool at ≥10–100M. Note: modern GPU-IVF with high `nlist` (or `IVF…_HNSW`) often matches IMI more simply. ([IMI paper](https://www.robots.ox.ac.uk/~vilem/cvpr2012.pdf); [revisiting inverted indices](https://arxiv.org/pdf/1802.02422))

---

## 11. GPU FAISS (`StandardGpuResources`) — batched, massive speedup

**How it works:** Move the index to GPU (`index_cpu_to_gpu`); the GPU parallelizes distance computation massively. Same API as CPU.

- **Speedup:** **~5–10× over the CPU implementation** on a single GPU (more for `Flat`/`IVFFlat` with large batches); with NVIDIA **cuVS** backend (FAISS ≥1.10), IVF-Flat search latency dropped up to **90%**.
- **Caveats that matter for the *per-query* metric:**
  - GPU shines with **large query batches**. At **batch=1**, kernel-launch overhead can make GPU *no faster or slower* than CPU; for small datasets `IVFFlat` on GPU can even be **slower than `Flat`**. Since PS-11 measures **avg time per query (batch=1)**, do not assume GPU wins — **measure**.
  - Best practice: if you must batch, batch all queries of one evaluation run together and divide total time by #queries.
- **Memory:** bounded by GPU VRAM; PQ indexes help.
- **API:**
  ```python
  res = faiss.StandardGpuResources()
  gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)   # device 0
  D, I = gpu_index.search(queries, k)                     # batch for max speedup
  ```
- **Verdict:** Great for **building** indexes fast and for **batched** evaluation throughput; for true single-query latency on small galleries, a tuned CPU index (FastScan/HNSW/Flat) is often equal or better and simpler. ([FAISS on GPU](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU); [cuVS blog](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/))

---

## 12. DiskANN / Vamana (billion-scale on SSD) — for scale far beyond PS-11

**How it works:** A graph index (**Vamana**) tuned for **smaller search radius** than HNSW/NSG so it minimizes random SSD reads; the full graph + full-precision vectors live on **SSD**, only compressed (PQ) vectors stay in RAM to guide traversal.

- **Performance:** on **1B-point SIFT1B**, **>5000 QPS, <3 ms mean latency, 95%+ 1-recall@1 on a 16-core machine with 64 GB RAM** — where FAISS/IVFOADC plateau ~50% recall at similar memory; ~**90% less RAM** than in-memory HNSW.
- **Query complexity:** graph traversal (≈logarithmic hops) bounded by SSD I/O.
- **When:** datasets too big for RAM (≥100M–1B). **Not needed for PS-11** (galleries are far smaller and fit in RAM), but the right answer if the archive scales to the full satellite-image-archive size. Available in `diskannpy`, Milvus, SQL Server 2025. ([DiskANN paper](https://suhasjs.github.io/files/diskann_neurips19.pdf); [Microsoft Research](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/))

---

## 13. Re-ranking — buy F1 back cheaply after a fast first stage

The pattern that wins this challenge: **fast/lossy first stage → re-rank only the top-K with something more accurate.** Latency added is `O(K·…)` with **K small (50–200)**, so it barely moves avg query time while measurably lifting F1@5/@10.

| Re-rank method | What it does | F1 / mAP effect | Added latency (top-K) | Verdict for PS-11 |
|---|---|---|---|---|
| **Exact-distance refine** (`IndexRefineFlat`, FastScan `RFlat`, ScaNN `reorder`) | Recompute *exact* float cosine on the K candidates from a compressed/ANN first stage; re-sort | Recovers most recall lost to PQ/quantization; near-Flat F1 | Tiny: K exact dot products (e.g. K=100, d=128 → ~12.8k mults) | **Always use** when first stage is compressed/approximate. Cheapest, safest win. |
| **k-reciprocal re-ranking** (Zhong CVPR'17) | Build k-reciprocal neighbor sets, encode as features, combine original + **Jaccard** distance; `d* = (1−λ)·d_Jaccard + λ·d_orig` | **Large**: on Market-1501 mAP **46.0 → 59.87 (+13.9 pts)**, rank-1 72.5 → 74.9 — *no training, no labels* | Moderate: building reciprocal sets is the cost; do it on **top-K only** (e.g. K≤200) to keep it cheap. Naïve full version is **O(N²)** (memory & time) — that's why you cap it to top-K | **Recommended** as the F1 booster. Params from paper: **k1≈20, k2≈6, λ≈0.3** (tune on val). Restrict to candidate pool. |
| **AQE / α-QE** (average / alpha query expansion) | Average the query with its top-`n` retrieved descriptors (α-QE weights by similarity^α), re-search once | Solid recall gain on instance retrieval; α-QE is the de-facto standard | One extra ANN search + an averaging | Cheap and effective; good if a second search is affordable. α≈1–3, n≈5–10. |
| **DBA** (database-side augmentation) | Offline: replace each gallery vector with a weighted avg of itself + its neighbors | Boosts recall; **zero query-time cost** (precomputed) | **0 at query time** (offline only) | Attractive: pay at index-build time, nothing per query. Combine with α-QE. |
| **Diffusion / graph re-ranking** (regularized random walk on the kNN manifold) | Propagate similarities over the neighborhood graph; finds true matches on the query's manifold | Strong on hard instance retrieval; can be decoupled into offline+online | Higher; an online matrix solve / GCN. GCN re-rank variants reach **~9.4 ms** on GPU | Use only if budget allows; usually overkill vs. k-reciprocal for this metric. |
| **Geometric / RANSAC verification** | Match local features between query & candidate, verify with a geometric model; count inliers | Highest precision for *instance/same-place*; great for SAR↔optical *co-located* pairs | Expensive per candidate (local-feature matching) → only top ~10–20 | Optional final precision pass for cross-modal co-location; too slow for large K. |

**Recommended cheap recipe:** *fast first stage → exact refine on top-100 → k-reciprocal on top-~20.* This adds well under a millisecond–few ms per query on typical galleries and reliably lifts F1@5/@10, especially for the harder **cross-modal** queries.
([k-reciprocal paper](https://arxiv.org/abs/1701.08398) & [numbers](https://www.emergentmind.com/papers/1701.08398); [efficient GCN re-rank](https://arxiv.org/pdf/2306.08792); [α-QE / DBA](https://arxiv.org/pdf/1811.00202); [FastScan refine](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)))

---

## 14. INDEX-FACTORY CHEATSHEET (per gallery size)

Assumes embeddings **L2-normalized**, `METRIC_INNER_PRODUCT` (cosine), `d` after optional PCA. `N` = gallery size. Use `4·√N … 16·√N` for `nlist`. **Always sweep `nprobe`/`efSearch` on a validation split until F1 plateaus, then back off to the latency budget.**

| Gallery `N` | Primary pick (best F1/latency) | `index_factory` string | Key runtime params | Memory / vec (d=128) | Notes |
|---|---|---|---|---|---|
| **~5k** | **Exact** | `"Flat"` (`IndexFlatIP`) | — | 512 B | Sub-ms batched. Don't over-engineer; this is your F1 ceiling. |
| **~5k** (alt, ultra-fast) | Binary hash + refine | ITQ→`IndexBinaryMultiHash` → refine `Flat` | radius/`k` for hash; rerank top-200 | 8–16 B (64–128 bit) + float for refine | Near-O(1) filter; refine restores F1. |
| **~50k** | **HNSW** (memory ok) | `"HNSW32,Flat"` | `efConstruction=80`, `efSearch=32–64` | ~1.0 KB | O(log N), recall 0.97+; no training. |
| **~50k** (alt) | **IVF exact lists** | `"IVF2048,Flat"` | `nprobe=8–32` | ~0.5 KB | √N≈224 → nlist 1024–4096; train on ≥30k. |
| **~50k** (alt, fastest CPU) | **FastScan + refine** | `"IVF2048,PQ32x4fsr,RFlat"` | `nprobe=16`, rerank `k_factor=10` | ~40 B + refine | Up to 1M QPS class; near-Flat F1 after refine. |
| **~500k** | **IVF + PQ/OPQ (+refine)** | `"OPQ32_128,IVF16384,PQ32"` | `nprobe=16–64`, add `IndexRefineFlat` | ~40 B | √N≈707 → nlist 4096–16384. Compact + fast. |
| **~500k** (recall-max) | **HNSW** if RAM allows | `"HNSW32,Flat"` | `efSearch=64–128` | ~1.0 KB | Highest recall; heavier RAM + build. |
| **~500k** (fastest CPU) | **FastScan** | `"IVF16384,PQ32x4fsr,RFlat"` | `nprobe=32`, `k_factor=10` | ~40 B | Top QPS-at-recall on CPU. |
| **~1M** | **IVF-HNSW + OPQ-PQ (+refine)** | `"OPQ32_128,IVF65536_HNSW32,PQ32"` | `nprobe=32–128`, refine top-100 | ~40 B | HNSW coarse quantizer removes large-nlist bottleneck. |
| **~1M** (recall-max, RAM ok) | **HNSW** | `"HNSW32,Flat"` | `efSearch=64–256` | ~1.0 KB | If it fits in RAM and build time is acceptable. |
| **≥100M / disk** | **DiskANN/Vamana** | (diskannpy / Milvus DISKANN) | search_L, beamwidth | PQ in RAM + graph on SSD | Out of PS-11 scope; for full-archive scale. |

**One shared cross-modal index (recommended):** build the chosen index over **all-modality** gallery embeddings; store a parallel `modality_id` array. For a query, retrieve top-(k+buffer), then **filter to the modality required by the evaluation case** (optical-only / SAR-only / or keep all for cross-modal). FAISS `IDSelector`/`SearchParameters` or post-filtering both work; post-filtering a slightly larger candidate set is simplest and keeps a single index in memory.

---

## 15. HOW TO MEASURE "average retrieval time per query" CORRECTLY

The metric is sensitive to methodology; report it honestly and reproducibly.

1. **Exclude index build/train/add from the timer.** Only time `search`. State build time separately.
2. **Warm up.** Run a few hundred–1000 throwaway queries first (page-in, JIT, caches, thread pool). FAISS benchmarks commonly use **~1000 warmup then 10,000 timed**. Discard warmup.
3. **Batch = 1 for the official "per query" number.** The metric says *per query*; the honest reading is single-query latency. Loop one query at a time, time each, report **mean and median (p50/p95)**.
   - If you *also* report a batched throughput number (queries/sec), label it clearly as throughput (FAISS is much faster batched because it parallelizes across queries).
4. **Pin threads deterministically.** Set `faiss.omp_set_num_threads(n)` (or `OMP_NUM_THREADS`); best when `n` = physical cores. For single-query latency, also try `n=1` to remove thread-launch overhead and report which you used.
5. **Use the same machine/CPU/GPU** for all variants you compare; record hardware, FAISS version, `d`, `N`, `nprobe`/`efSearch`, and recall/F1 at that operating point (latency without the accuracy it buys is meaningless).
6. **Include the full query path** the challenge counts: embedding the query (if counted) + ANN search + any re-ranking. Re-ranking time must be inside the measured window.
7. **Repeat & average** over the whole query set (all same-modal and cross-modal queries); report per-case if they differ.

```python
import time, numpy as np, faiss
faiss.omp_set_num_threads(1)              # single-query latency; or =physical cores
# warmup
for q in warmup_queries: index.search(q.reshape(1, -1), k)
# timed, batch=1
lat = []
for q in eval_queries:
    t0 = time.perf_counter()
    D, I = index.search(q.reshape(1, -1), k)     # + reranking here if counted
    lat.append((time.perf_counter() - t0) * 1e3) # ms
lat = np.array(lat)
print(f"avg {lat.mean():.3f} ms | p50 {np.percentile(lat,50):.3f} | p95 {np.percentile(lat,95):.3f}")
```
([FAISS threads](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls); [make FAISS faster](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster))

---

## 16. FINAL RECOMMENDED DESIGN

### 16a. Architecture: ONE shared cross-modal index + L2-norm + PCA
- **Single shared FAISS index over all-modality embeddings** (optical + multispectral + SAR), because alignment already places them in one space — a single kNN then naturally returns mixed-modality neighbors, which is exactly cross-modal retrieval. Keep a `modality_id` side array to filter per evaluation case. (Building separate per-modality indexes is only worth it if you must *guarantee* modality isolation or shard for scale; for PS-11 it adds complexity and duplicates infrastructure for no F1 benefit.)
- **L2-normalize everything → inner product = cosine** (`faiss.normalize_L2`, `METRIC_INNER_PRODUCT`).
- **PCA-whiten to 128 (or 64) dims** (`PCAR128`) if the backbone emits ≥512-d — cheaper distances, smaller index, usually negligible F1 loss; validate.

### 16b. The hybrid "near-O(1) filter → exact re-rank" pipeline (best F1-vs-latency)
1. **Coarse filter (near-O(1)-ish):** learn **binary codes** (CSQ/DPSH hashing head if you train one; else **ITQ** post-hoc on the float embeddings) and search them with **`IndexBinaryMultiHash` (MIH)** — hash-table lookup, effectively constant-time per query for short codes. Fetch a **candidate pool of ~200–500** by Hamming distance.
   - *If you prefer an all-float stack:* substitute a **FastScan IVF** (`IVF…,PQ32x4fsr`) or **HNSW** first stage — both sublinear and extremely fast — to get the same candidate pool.
2. **Exact re-rank (precision):** recompute **exact float cosine** on just the candidate pool (`IndexRefineFlat` or a manual matmul) and take the top-100. This restores Flat-level F1 at trivial cost.
3. **k-reciprocal re-rank (F1 boost):** apply k-reciprocal encoding on the **top ~20–50** only (k1≈20, k2≈6, λ≈0.3). Adds little latency, lifts F1@5/@10 — most valuable on the harder cross-modal queries.
4. **(Optional)** geometric/RANSAC verification on the final top-10 for cross-modal *co-located* precision, if the budget remains.

This gives **hash-table/ANN speed for the 99% of work (pruning N→a few hundred)** and **exact-search accuracy for the 1% that determines F1.**

### 16c. Concrete `index_factory` by gallery size (cosine, normalized)
- **~5k:** `"Flat"` — exact, sub-ms, best F1. (Optional ultra-fast: ITQ-64 → `IndexBinaryMultiHash` → refine `Flat`.)
- **~50k:** `"HNSW32,Flat"` (`efSearch=32–64`) **or** fastest-CPU `"IVF2048,PQ32x4fsr,RFlat"` (`nprobe=16`, `k_factor=10`).
- **~500k:** `"OPQ32_128,IVF16384,PQ32"` + `IndexRefineFlat` (`nprobe=16–64`) **or** fastest-CPU `"IVF16384,PQ32x4fsr,RFlat"` (`nprobe=32`).
- **~1M:** `"OPQ32_128,IVF65536_HNSW32,PQ32"` + refine top-100 (`nprobe=32–128`). Recall-max alt if RAM permits: `"HNSW32,Flat"` (`efSearch=64–256`).
- **≥100M / disk-bound:** DiskANN/Vamana (out of scope here).

### 16d. Measurement discipline
Time `search` only; warmup ~1000; **report per-query mean + p50/p95 at batch=1**; fix `omp` threads; always pair latency with the F1/recall it achieves; include re-ranking inside the timer. ([§15](#15-how-to-measure-average-retrieval-time-per-query-correctly))

---

## 17. Copy-pasteable FAISS snippets

```python
import faiss, numpy as np

d = 128                       # embedding dim (after optional PCA)
gallery = np.load("gallery_emb.npy").astype("float32")   # (N, d)
queries = np.load("query_emb.npy").astype("float32")     # (nq, d)
modality_id = np.load("gallery_modality.npy")            # (N,) e.g. 0=opt,1=ms,2=sar
faiss.normalize_L2(gallery); faiss.normalize_L2(queries) # cosine via inner product
k = 10

# ---------- 0) Exact baseline (small gallery, F1 ceiling) ----------
flat = faiss.IndexFlatIP(d); flat.add(gallery)
D, I = flat.search(queries, k)

# ---------- 1) PCA pre-transform + IVF (medium gallery) ----------
# (skip PCA if d already small; PCAR also whitens/rotates -> helps PQ)
index = faiss.index_factory(d, "PCAR128,IVF4096,Flat", faiss.METRIC_INNER_PRODUCT)
index.train(gallery); index.add(gallery); index.nprobe = 16
D, I = index.search(queries, k)

# ---------- 2) HNSW (high recall, O(log N), no training) ----------
hnsw = faiss.index_factory(d, "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 80; hnsw.add(gallery)
hnsw.hnsw.efSearch = 64
D, I = hnsw.search(queries, k)

# ---------- 3) Compact OPQ+IVFPQ + EXACT refine (large gallery) ----------
base = faiss.index_factory(d, "OPQ32_128,IVF16384,PQ32", faiss.METRIC_INNER_PRODUCT)
base.train(gallery); base.add(gallery)
faiss.extract_index_ivf(base).nprobe = 32
refine = faiss.IndexRefineFlat(base)          # re-rank candidates with exact distances
refine.k_factor = 10                          # fetch k*10, refine to k
D, I = refine.search(queries, k)

# ---------- 4) FastScan IVF + refine (fastest CPU operating point) ----------
fs = faiss.index_factory(d, "IVF16384,PQ32x4fsr,RFlat", faiss.METRIC_INNER_PRODUCT)
fs.train(gallery); fs.add(gallery)
faiss.extract_index_ivf(fs).nprobe = 32
D, I = fs.search(queries, k)

# ---------- 5) Binary hashing (ITQ) + Multi-Index Hashing (near-O(1)) ----------
nbits = 64
vt = faiss.index_factory(d, f"PCAR{nbits},ITQ{nbits},LSHt")  # produce binary codes
vt.train(gallery)
codes  = vt.sa_encode(gallery)                # packed uint8, nbits/8 per vector
qcodes = vt.sa_encode(queries)
bindex = faiss.IndexBinaryMultiHash(nbits, 4, nbits // 4)    # MIH-style hash tables
bindex.add(codes)
Dham, Icand = bindex.search(qcodes, 200)      # near-O(1) Hamming filter -> 200 cands
# exact float re-rank of the candidate pool (cosine) -> final top-k
for qi in range(len(queries)):
    cand = Icand[qi]; cand = cand[cand >= 0]
    sims = gallery[cand] @ queries[qi]        # exact inner product
    topk = cand[np.argsort(-sims)[:k]]

# ---------- 6) GPU (batched evaluation throughput) ----------
res = faiss.StandardGpuResources()
gpu = faiss.index_cpu_to_gpu(res, 0, flat)    # any CPU index
D, I = gpu.search(queries, k)                 # batch all queries for max speedup

# ---------- Cross-modal: one shared index, filter by modality ----------
D, I = index.search(queries, k * 4)           # over-fetch from the SHARED index
for qi in range(len(queries)):
    want = SAR                                # desired target modality for this case
    rows = [j for j in I[qi] if modality_id[j] == want][:k]
```

---

## 18. Sources
- FAISS — Guidelines to choose an index: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- FAISS — Faiss indexes (types/params): https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- FAISS — Binary indexes: https://github.com/facebookresearch/faiss/wiki/Binary-indexes
- FAISS — Binary hashing index benchmark: https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark
- FAISS — Fast accumulation of PQ/AQ codes (FastScan): https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)
- FAISS — MetricType and distances: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
- FAISS — Threads and asynchronous calls: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
- FAISS — How to make Faiss run faster: https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster
- FAISS — Faiss on the GPU: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
- NVIDIA cuVS in FAISS: https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/
- The FAISS library (paper): https://arxiv.org/pdf/2401.08281
- Pinecone — Vector indexes / IVF / PQ / HNSW series: https://www.pinecone.io/learn/series/faiss/vector-indexes/ , https://www.pinecone.io/learn/series/faiss/product-quantization/ , https://www.pinecone.io/learn/series/faiss/hnsw/
- IVFPQ+HNSW billion-scale: https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/
- ScaNN — Anisotropic VQ (paper): http://proceedings.mlr.press/v119/guo20h/guo20h.pdf ; arXiv: https://arxiv.org/pdf/1908.10396
- ScaNN — Google blog: https://research.google/blog/announcing-scann-efficient-vector-similarity-search/
- FAISS vs ScaNN comparative study: https://arxiv.org/html/2507.16978v1
- ScaNN vs 4-bit PQ: https://medium.com/@kumon/similarity-search-scann-and-4-bit-pq-ab98766b32bd
- Multi-Index Hashing (Norouzi, PAMI 2014): https://www.cs.toronto.edu/~fleet/research/Papers/MIH-pami2014.pdf ; code: https://github.com/norouzi/mih
- Inverted Multi-Index (Babenko & Lempitsky): https://www.robots.ox.ac.uk/~vilem/cvpr2012.pdf ; Revisiting inverted indices: https://arxiv.org/pdf/1802.02422
- ITQ (Gong & Lazebnik): https://slazebni.cs.illinois.edu/publications/ITQ.pdf
- DPSH (deep pairwise-supervised hashing): https://arxiv.org/abs/1511.03855
- CSQ (Central Similarity Quantization, CVPR 2020): https://arxiv.org/abs/1908.00347 ; code: https://github.com/swuxyj/DeepHash-pytorch/blob/master/CSQ.py
- Deep text/semantic hashing survey: https://arxiv.org/pdf/2510.27232
- Deep hash RS retrieval w/ semantic cues: https://www.mdpi.com/2072-4292/14/24/6358 ; Deep contrastive SSL hashing for RS: https://www.mdpi.com/2072-4292/14/15/3643
- k-reciprocal re-ranking (Zhong CVPR'17): https://arxiv.org/abs/1701.08398 ; numbers: https://www.emergentmind.com/papers/1701.08398
- Efficient GCN-based re-ranking: https://arxiv.org/pdf/2306.08792
- α-QE / DBA (attention-aware GeM): https://arxiv.org/pdf/1811.00202 ; attention-based QE learning: https://arxiv.org/pdf/2007.08019
- DiskANN/Vamana (NeurIPS'19): https://suhasjs.github.io/files/diskann_neurips19.pdf ; MSR: https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/
- FAISS↔cosine (normalize_L2) practice: https://github.com/facebookresearch/faiss/issues/1119 ; PCA in FAISS: https://github.com/facebookresearch/faiss/issues/400
- Dimensionality reduction for retrieval speed/memory: https://www.sciencedirect.com/science/article/pii/S2666912922000241
- Cross-modal RS datasets (context): 3MOS https://link.springer.com/article/10.1007/s44267-025-00091-0 ; SOMA-1M https://arxiv.org/pdf/2602.05480
