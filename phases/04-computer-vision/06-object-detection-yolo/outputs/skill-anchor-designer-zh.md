---
name: skill-anchor-designer
description: Given a dataset of ground-truth boxes, run k-means on (w, h) and return anchor (锚框) sets per FPN level plus coverage statistics
version: 1.0.0
phase: 4
lesson: 6
tags: [computer-vision, detection, anchors, kmeans]
---

# Anchor Designer

Anchors are the single most dataset-specific hyperparameter in an anchor-based detector. Default COCO anchors underperform on cell-culture images, satellite tiles, or small-object surveillance. This skill derives anchors that actually match the target data.

## When to use

- Before a first 训练 run on a new dataset.
- When recall (召回率) on very small or very large objects is weak on an otherwise healthy model.
- After a major dataset expansion where box size distribution may have shifted.

## Inputs

- `boxes`: numpy array of shape (N, 4) in either `(cx, cy, w, h)` or `(x1, y1, x2, y2)` format; at least 1000 positive boxes recommended.
- `num_anchors_per_level`: usually 3.
- `num_fpn_levels`: usually 3 (P3, P4, P5) or 4.
- `input_size`: 训练-resolution HxW.
- Optional `strides`: per-level strides; when omitted, take the first `num_fpn_levels` entries of `[8, 16, 32, 64]`. Pass a longer or shorter array explicitly if the detector's FPN has different strides.

## Steps

1. **Normalise boxes** to `(w, h)` pairs in pixel (像素) units at `input_size`. Drop any with w or h < 2 pixels.

2. **运行 k-means** on `(w, h)` pairs, with `k = num_anchors_per_level * num_fpn_levels`. Use `1 - IoU (交并比)(box, cluster)` as the distance function, not Euclidean distance — Euclidean on `(w, h)` collapses thin tall boxes and square boxes together. All boxes contribute equally (unweighted); if you have a class-imbalanced dataset and want larger-box recall (召回率), repeat rare-class boxes in the input array rather than passing a weight vector.

3. **Sort clusters by area** ascending. Split into `num_fpn_levels` groups of `num_anchors_per_level`. Smallest areas go to the highest-resolution level (smallest stride (步幅)).

4. **Compute coverage statistics** per level:
   - `median IoU (交并比)` of each ground-truth box to its best anchor (锚框) at that level.
   - `recall (召回率)@IoU (交并比)=0.5` — percentage of boxes whose best anchor (锚框) has IoU >= 0.5.
   - `area coverage` — fraction of boxes whose area falls within `[anchor_min_area / 4, anchor_max_area * 4]` of the level.

5. **Report per-level anchors** and flag levels where `recall (召回率)@IoU (交并比)=0.5 < 0.9`; that level's anchors do not match the data well and should be retuned or the number of anchors per level increased.

## Report format

```
[anchor-designer]
  total boxes:         <N>
  clusters:            <k>
  distance metric:     1 - IoU

[level P3  stride=8]
  anchors (w, h):      [(A, B), (C, D), (E, F)]
  median IoU:          <X>
  recall@IoU=0.5:      <X>
  coverage:            <X>
  flag:                ok | retune

[level P4  stride=16]
  ...

[summary]
  overall recall@IoU=0.5: <X>
  smallest anchor:        <w x h>
  largest anchor:         <w x h>
  recommendation:         <one sentence if any level flagged>
```

## Rules

- Always use IoU-based distance; Euclidean k-means produces visually reasonable but empirically worse anchors.
- Sort clusters by area, then assign to levels in ascending order.
- When `num_anchors_per_level = 1`, skip k-means entirely: split boxes into `num_fpn_levels` bins by area quantile (e.g. terciles for 3 levels), and set each level's anchor (锚框) to the per-bin median (w, h). This is more robust than running k-means with `k = num_fpn_levels` on small datasets.
- Never output negative anchor (锚框) dimensions; clamp at 1.
- If the dataset has < 200 boxes, warn the user that anchor (锚框) search is unreliable and recommend using default COCO anchors plus more 训练 data.
