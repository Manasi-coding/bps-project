# Point Cloud Reconstruction Audit — `obj_sphere`

## Objective

Audit why the reconstructed point cloud full width is ~0.075 m while the SDF ground-truth diameter is 0.096 m. Diagnostic only — no code changes or fixes.

## Evidence collected

- Mask file: `data/masks/world1_primitives/obj_sphere_mask.npy` (mask pixels = 2982)
- Boundary file: `data/masks/world1_primitives/obj_sphere_boundary.npy` (boundary pixels = 303, all valid)
- Depth map: `depth_maps/world1_primitives_depth.npy`
- RGB image: `rgb_images/world1_primitives_rgb.png` (present)
- Ground-truth: `ground_truth/world1_primitives_pose.json` (gt width = 0.096 m)
- Camera intrinsics: no saved intrinsics found; diagnostics used default intrinsics `fx=fy=570.3, cx=319.5, cy=239.5` (noted in code as common pipeline defaults).

Computed numbers (from the masked depth pixels and standard pinhole backprojection):

- Masked pixels (after finite-range check): 2982
- Median depth: 0.67072254 m
- MAD: 0.01033971 m
- Depth gate (median + max(6*MAD, 0.03)): 0.73276079 m (no pixels removed by gate)
- Depth percentiles (5%, 95%): 0.65531231 m, 0.70090692 m

Raw point cloud (backprojected masked pixels):

- Number of valid 3D points: 2982
- X min, X max: -0.14444975 m, -0.06738059 m
- Y min, Y max: 0.23144833 m, 0.28814933 m
- Z min, Z max: 0.65409893 m, 0.72001171 m
- X/Y/Z extents (max - min): 0.07706916 m, 0.05670100 m, 0.06591278 m
- Centroid (x,y,z): [-0.10222822, 0.25424197, 0.67350521]
- Boundary pixel depths: min=0.6697843 m, max=0.7200117 m (all 303 boundary pixels have valid depth)

PCA-projected extents (for reference, used later by estimator):

- Full PCA-projected width (u_max − u_min): 0.07515702 m
- Full PCA-projected height (v_max − v_min): 0.07904032 m
- 5th–95th percentile widths used by estimator: 0.05535453 m (u), 0.06380617 m (v)

## Confirmed observations

- The semantic mask covers the visible sphere pixels in the provided depth map: mask pixel count is substantial (2982) and all 303 boundary pixels contain valid depth.
- The median/MAD depth gate does not remove any masked pixels for this scene (kept=2982, dropped=0).
- Backprojection used the standard pinhole formula (x = (u-cx)*z/fx, y = (v-cy)*z/fy) with the pipeline's typical intrinsics; results produce a coherent spherical cloud (centroid and covariances consistent with a near-isotropic object).
- The raw cloud's full X extent (~0.07707 m) is already ~0.019 m smaller than the SDF diameter (0.096 m). This discrepancy is present before PCA or percentile trimming.

## Requires verification (diagnostic follow-ups — data collection only)

- Confirm camera intrinsics used by the live pipeline (camera_info) match the defaults assumed here. If the runtime intrinsics differ, re-run the backprojection with the exact `fx, fy, cx, cy` to verify scale.
- Visually overlay `obj_sphere_mask.png` on `rgb_images/world1_primitives_rgb.png` and the depth image crop to confirm mask alignment at sub-pixel level and ensure no consistent inward offset of mask boundary.
- Inspect raw depth-image pre-processing steps (smoothing, hole-filling) used when producing `world1_primitives_depth.npy` to see if edges are smoothed inward.
- Confirm SDF export for `obj_sphere` indeed encodes diameter 0.096 m (already observed in `ground_truth` file, but re-verify source SDF if needed).

## Most likely stage where the missing ~21 mm is introduced

Based on the evidence, the missing ~21 mm appears before PCA/percentile estimation — specifically within the reconstructed point cloud itself. The raw backprojected cloud's full X extent (~0.077 m) already accounts for the majority of the discrepancy vs. the 0.096 m ground-truth. The masking and depth-gating stages did not remove boundary points in this sample; therefore the discrepancy likely originates from the sensor/depth-image measurement or the backprojection inputs (camera intrinsics or depth preprocessing), rather than the PCA or percentile trimming stages.

---

Files used for diagnostics and verification commands are the same as listed above. This report contains measured quantities only; no code changes or parameter recommendations are included.