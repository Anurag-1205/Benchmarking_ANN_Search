# Hypotheses

Rule from the assignment: each hypothesis is committed to git **before** the experiment it covers is run, and the
git history must show that order. Do not edit a committed hypothesis after seeing results; if you want to comment
on it, add a dated "After the results" note below it in a later commit.

---

## Part 3 - experiment: Query difficulty on GloVe-100

**Status:** written 2026-10-23, before the experiment was run.

**Setup.** For each of the 5,000 report queries, relative contrast RC = (mean distance from the query to every
base vector) / (distance to its nearest neighbour), computed exactly in chunks. The queries are split into four
equal bins by RC quartile (bin 1 = lowest RC = hardest). Each index is evaluated at fixed operating points chosen
on the tune queries: for HNSW and IVF-Flat, the fastest Part 2 setting that reaches recall@10 >= 0.90 and >= 0.95
on the tune half; for IVF-PQ, its highest-recall setting (it never reaches 0.90). Measured: mean recall@10 per bin
on the report half.

- **Prediction:** recall falls monotonically from the easiest bin (highest RC) to the hardest one, for every index.
  At the 0.95 operating point the hardest bin is at least 0.05 below the easiest one. The drop is largest for
  IVF-PQ, then IVF-Flat, and smallest for HNSW.
- **Mechanism:** a low RC means the true neighbours are barely closer than everything else, so every approximation
  has less margin. HNSW's greedy walk steers by differences in distance; with near-ties its candidate list
  (efSearch) fills with almost-as-close points and stops before it has reached all true neighbours. For IVF-Flat,
  the true neighbours of a low-RC query lie far from the query relative to the cell size, so they are spread over
  more Voronoi cells and a fixed nprobe covers fewer of them. IVF-PQ adds a quantisation error of roughly constant
  size, which reorders near-ties exactly when the margin is small.
- **What would contradict it:** for some index, recall in the hardest bin within 0.02 of the easiest bin; a
  non-monotone pattern across the bins; or HNSW dropping more than IVF-Flat.

---

## Part 4 - experiment: Hard geometry - intrinsic dimension

**Status:** written 2026-10-23, before the experiment was run.

**Setup.** Seeded synthetic data: 200,000 base vectors and 2,000 queries (1,000 tune / 1,000 report) drawn from a
standard Gaussian in m dimensions, zero-padded to 128 dimensions and rotated by one fixed random rotation. Only m,
the intrinsic dimension, changes; the stored dimension stays 128, so memory and the cost of one distance are the
same in every condition. Control: m = 8. Settings: m = 16, 32, 64, 128. Two indexes with fixed build settings:
HNSW (M = 16, efConstruction = 200) and IVF-Flat (nlist = 512), each swept over efSearch / nprobe as in Part 2.
Measured: the relative contrast of each dataset, recall@10 at fixed search settings, and the single-query QPS
needed to reach recall 0.95, compared with exact search.

- **Prediction:** as m grows, relative contrast falls towards 1, recall at any fixed efSearch or nprobe falls, and
  the QPS at recall 0.95 drops by more than 10x between m = 8 and m = 128 for both indexes. IVF-Flat degrades
  faster than HNSW: at m = 128 it has to scan a large share of the lists and is no faster than exact search.
- **Mechanism:** in high intrinsic dimension, distances concentrate: a query's nearest neighbours are only slightly
  closer than a random point. k-means cells then no longer contain neighbourhoods, so a query's true neighbours
  are scattered over many cells and nprobe has to approach nlist. In HNSW the edges are chosen among nearly
  equidistant candidates and give the greedy walk little direction, so efSearch must grow a lot before the walk
  stops missing neighbours.
- **What would contradict it:** QPS at recall 0.95 at m = 128 within 3x of the control; recall at fixed settings
  not falling as m grows; or HNSW degrading faster than IVF-Flat.
