# Sphere Estimation Audit

## Objective

Audit remaining estimation error for `obj_sphere` (diagnostic only — no code changes).

## Current observation

- Ground-truth diameter: 0.096 m
- Recorded estimated width (pipeline logs): 0.077 m (error −0.019 m)

## Summary conclusion

- Depth filtering (median/MAD depth gate) did not remove valid object points — all object-mask pixels survived the gate.
- The reconstructed point cloud (from masked depth pixels) shows a full projected width of ~0.075 m.
- The implemented percentile trimming (5th–95th) reduces the measured width to ~0.055 m — a reduction of ~0.020 m, which matches the observed underestimation magnitude.
- Therefore the primary contributor to the pipeline's underestimation is the percentile trimming; a secondary contributor is that the point cloud's full measured extent (~0.075 m) is already ~0.021 m smaller than the SDF ground-truth diameter (possible causes: sensor/registration limits, mask not including extreme boundary pixels, or object partial occlusion).

## Collected evidence (measurements)

### Depth filtering / median-MAD gate

- Mask pixels (object): 2982
- Valid pixels after finite-range filter (>0.05 & <5.0 m): 2982
- Median depth: 0.6707225441932678 m
- MAD: 0.010339707136154175 m
- Depth gate (median + max(6*MAD, 0.03)): 0.7327607870101929 m
- Depth percentiles (5th, 95th): 0.6553123086690903 m, 0.7009069204330445 m
- Kept after depth gate: 2982 (dropped 0)

### Point cloud reconstruction

- Points used (after mask + valid depth): 2982
- Depth span (z_max - z_min): 0.06591278314590454 m
- Observed shape: cohesive spherical cloud with no large background leakage or removals by gate.

### Percentile extent estimation (PCA-projected)

- Full projected extents (principal-axis coordinates):
  - u (min, max): −0.027936536740167783 m, 0.04722048648135306 m
  - v (min, max): −0.04216456295649264 m, 0.03687575514653697 m
  - Full width (u_max − u_min): 0.07515702322152085 m
  - Full height (v_max − v_min): 0.07904031810302961 m
- 5th–95th percentiles (used by estimator):
  - u (5%, 95%): −0.024129343686884692 m, 0.031225190005600183 m
  - v (5%, 95%): −0.03254908057204297 m, 0.031257094331816795 m
  - Percentile width (u_95 − u_5): 0.05535453369248487 m
  - Percentile height (v_95 − v_5): 0.06380617490385976 m
- Effect: percentile trimming reduces measured width by ~0.0198 m (0.07516 → 0.05535).

### PCA diagnostics

- Covariance eigenvalues: [7.63434359e-05, 2.96273553e-04, 4.00209954e-04]
- Principal axes appear reasonable for a near-isotropic spherical surface (no obvious axis misalignment).

## Interpretation and recommended next diagnostics (non-invasive)

1. Percentile trimming is the most likely immediate cause of the observed underestimation because it removes roughly the same magnitude (~0.02 m).
2. The point cloud's full measured width (~0.075 m) is still smaller than the ground truth (0.096 m). Investigate why the cloud misses ~0.02 m of radius:
   - Visualize `obj_sphere_mask.png` and overlay it on the depth image crop to confirm mask includes object's visual rim.
   - Visualize the boundary mask `obj_sphere_boundary.npy` overlayed on the depth image to confirm boundary pixels have finite depth and are included (boundary pixels: 303, all valid in the sampled depth map).
   - Examine raw depth image values at boundary pixels to check for systematic bias (e.g., smoothing, quantization, or sensor dropouts near grazing angles).
3. Recompute extents using full min/max (no percentile trimming) and/or more permissive percentiles (e.g., 1%–99%) for a short A/B diagnostic to confirm how much percentile choice changes the evaluation (do not change production code — run as an offline diagnostic).
4. If full cloud extent remains ~0.075 m vs GT 0.096 m, check SDF geometry scaling/export (confirm `export_pose.py` extracted diameter correctly) and camera intrinsics/registration.

## Files & commands used for diagnostics

- Masks: `data/masks/world1_primitives/obj_sphere_mask.npy` and `.../obj_sphere_boundary.npy`
- Depth map: `depth_maps/world1_primitives_depth.npy`
- Estimator inspected: `scripts/grasp_estimator.py`

Quick commands used (example):

```bash
python3 -c "import numpy as np; m=np.load('data/masks/world1_primitives/obj_sphere_mask.npy'); d=np.load('depth_maps/world1_primitives_depth.npy'); print(m.sum(), np.nanmin(d[m>0]), np.nanmax(d[m>0]))"
```

---

If you want, I can produce the suggested visualizations (overlay mask on depth crop, plot boundary pixel depths, or run a one-off extent computation with different percentile settings) and add results to this audit file.