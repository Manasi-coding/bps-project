# Grasp Estimator Audit

## Objective
Summarize the code audit for `scripts/grasp_estimator.py`, focusing on where estimation error may be introduced in the semantic mask, depth filtering, PCA-based extent estimation, and ground-truth comparison stages.

## Key Findings

### 1. Semantic Mask Alignment
- Current assumptions
  - The semantic mask file exists at `semantic_masks_dir/<world_name>/<object_name>_mask.npy`.
  - The mask is aligned pixel-for-pixel with the depth image if its shape matches the depth image shape.
- Evidence available
  - Code checks only `mask.shape == depth.shape` at lines 457-462.
  - No additional validation is performed on mask pixel values or coordinate registration.
- What still needs verification
  - Whether mask pixel coordinates correspond to the same image frame and origin as the depth image.
  - Whether the mask contains the expected binary/nonzero object region and no misaligned background.

### 2. Depth Filtering
- Median/MAD depth gate
  - The filtered depth values `zs` are computed from masked pixels, then pruned by `median_z + max(6*m ad, 0.03)` at lines 482-487.
  - Invalid or out-of-range depths are removed first using `np.isfinite(zs) & (zs > 0.05) & (zs < 5.0)`.
- What evidence is missing
  - Whether the median and MAD statistics are dominated by object points or by mask contamination.
  - How many points are removed by the depth gate and whether the surviving cloud still represents the object.
- Diagnostics required
  - counts before/after filtering, `median_z`, `mad`, `depth_gate`, and depth percentiles.

### 3. PCA Axis Interpretation
- Current assumptions about eigenvectors
  - The code assumes the two largest-variance principal components correspond to object width and height (lines 691-692).
  - It ignores the smallest eigenvector and uses projections onto `eigvecs[:,1]` and `eigvecs[:,2]` as width/height axes.
- Why this needs verification
  - The eigenvector ordering from `np.linalg.eigh(cov)` is ascending by eigenvalue, so the axis selection must be verified against the object geometry.
  - The estimator does not inspect whether the chosen PCA axes align with the expected physical object dimensions.
- Diagnostics required
  - eigenvalues and eigenvectors from the covariance matrix, plus projected extents along each axis.

### 4. Ground Truth Consistency
- Whether exported width/height/depth are guaranteed to match the estimator's definitions
  - The code loads GT width/height/depth from JSON using `export_pose.py` output at lines 365-377.
  - It then compares `estimated_width` and `estimated_height` directly to `gt_width` and `gt_height` at lines 520-535.
- What should be verified
  - Whether the GT width/height axes use the same conventions as the estimator’s PCA-based width/height.
  - Whether `gt_depth` is comparable to the estimator's raw `zs.max() - zs.min()` depth measure.

### 5. Point Cloud Inspection
- Why the reconstructed point cloud should be inspected
  - The object reconstruction is produced by backprojecting masked depth pixels to 3D in `_estimate_dimensions()` at lines 683-686.
  - Any mask misalignment, bad depths, or camera intrinsics mismatch will appear in the point cloud.
- What properties should be checked (extent, outliers, missing regions)
  - point count, finiteness, min/max per axis, and overall spread
  - whether the cloud is planar or volumetric
  - whether `depth_out` matches the z-axis spread of the same cloud

## Audit Priority
1. Semantic Mask Alignment — highest priority because the entire pipeline depends on correct pixel registration, and the code only validates shape.
2. Depth Filtering — requires verification because the median/MAD gate can remove valid object points or preserve background if the mask is impure.
3. PCA Axis Interpretation — important because estimated width/height depend on assumed PCA axes, and the code does not validate axis meaning.
4. Ground Truth Consistency — direct comparison is made, but only valid if GT axes and estimator axes are aligned.
5. Point Cloud Inspection — useful to verify the raw reconstructed object geometry after mask and depth processing.
