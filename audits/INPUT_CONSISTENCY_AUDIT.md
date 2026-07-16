# Input Consistency Audit — `obj_sphere`

## Objective

Verify whether the offline diagnostics and live runtime use the same input data for `obj_sphere` and identify the first stage where they diverge. This is a data-audit only; no code changes or fixes.

## Offline (reproduction) values

These values were measured from the saved offline assets in the repo (`depth_maps/world1_primitives_depth.npy`, `data/masks/world1_primitives/obj_sphere_mask.npy`) and the offline backprojection used in earlier audits.

- Mask pixel count: 2982
- Valid depth pixel count (finite and 0.05 < z < 5.0): 2982
- Pixels after depth filtering (median/MAD depth gate): 2982
- Median depth: 0.6707225441932678 m
- MAD: 0.010339707136154175 m
- Depth gate (median + max(6*MAD, 0.03)): 0.7327607870101929 m
- Depth percentiles (5%, 95%): 0.6553123086690903 m, 0.7009069204330445 m
- Camera intrinsics assumed (offline): fx=570.3, fy=570.3, cx=319.5, cy=239.5
- Depth image resolution: (480, 640)
- Depth image dtype: float32 (from saved .npy)
- Depth image timestamp/header: not present in saved .npy (no ROS header)
- Point-cloud centroid (offline backprojected using assumed intrinsics): [-0.10222822, 0.25424197, 0.67350521]

Notes: the offline audit also computed PCA projected full span ≈ 0.07515702 m and 5–95 percentile span ≈ 0.05535453 m; those are downstream of these inputs.

---

## What to capture from the live runtime (minimal, non-invasive)

Run the capture scripts created in `LIVE_RUNTIME_AUDIT.md` (or the short commands below) while the pipeline is running and `grasp_estimator.py` is evaluating `obj_sphere`:

1) Save `live_camera_info.json` and `live_depth.npy` (one frame) — run the provided `save_live_camera_depth.py`:

```bash
python3 save_live_camera_depth.py
```

This writes:
- `live_camera_info.json` (width, height, `K`, `D`, distortion model)
- `live_depth.npy` (depth frame as numpy array)
- `live_depth_header.json` (stamp, frame_id)

2) Reproject the captured live depth frame using the estimator's backprojection (use `reproject_live.py`):

```bash
python3 reproject_live.py
```

This writes:
- `live_reprojection_stats.json` (n_points, min_xyz, max_xyz, extents, centroid)
- `live_reprojection_points.npy`

3) (Optional) If you instrumented `grasp_estimator.py` for TRACE logs, capture the estimator log lines (they include `TRACE INPUT obj_sphere` and `TRACE COUNTS obj_sphere`).

---

## Placeholders for runtime values (paste outputs here after running the capture)

- Mask pixel count (runtime):
- Valid depth pixel count (runtime):
- Pixels after depth filtering (runtime):
- Median depth (runtime):
- MAD (runtime):
- Depth gate (runtime):
- Depth percentiles (5%,95%) (runtime):
- Camera intrinsics (runtime): fx= , fy= , cx= , cy= 
- Depth image resolution (runtime):
- Depth dtype (runtime):
- Depth header/timestamp (runtime):
- Point-cloud centroid (runtime):

Paste the contents of `live_camera_info.json`, `live_depth_header.json`, and `live_reprojection_stats.json` into this file or attach them; I will compute differences and fill the next section.

---

## How differences will be evaluated

I will compare offline vs runtime for each item (exact numeric difference and relative percent where appropriate). The first stage where values differ will be reported as the "first point of divergence" (e.g., different intrinsics → reprojection scale mismatch; different depth frame → pixel/depth mismatch; different mask → mask mismatch).

## First point of divergence (to be filled after runtime capture)


## Runtime vs Offline differences (to be filled after runtime capture)


---

When you paste the three JSON outputs (`live_camera_info.json`, `live_depth_header.json`, `live_reprojection_stats.json`) or the raw TRACE lines from `grasp_estimator.py`, I'll compute and record the differences and identify the first divergence point in `INPUT_CONSISTENCY_AUDIT.md`.