# Runtime Pipeline Consistency Audit — `obj_sphere`

## Objective

Verify whether the offline diagnostics faithfully reproduce the live runtime pipeline used by `grasp_estimator.py` for `obj_sphere`. This is a verification-only audit; no code changes or fixes are proposed.

## Evidence collected

- Runtime code paths inspected:
  - `scripts/grasp_estimator.py` (subscribes to `/depth_model/depth_image` and `/camera_info`, backprojects masked pixels using `x = (u-cx)*z/fx`, `y = (v-cy)*z/fy`, and uses the depth value directly)
  - `scripts/depth_to_pointcloud.py` (subscribes to `/depth_model/depth_image` and `/camera_info`, converts reported range `r` into perpendicular Z-depth via `Z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)`, then applies a world-frame transform before publishing a PointCloud2)
- Files used for offline reproduction (available in repository):
  - `depth_maps/world1_primitives_depth.npy` (depth image used in earlier offline audit)
  - `data/masks/world1_primitives/obj_sphere_mask.npy` (semantic mask)
  - `ground_truth/world1_primitives_pose.json` (gt width = 0.096 m)
  - `rgb_images/world1_primitives_rgb.png` (RGB image present)
- Intrinsics: no saved CameraInfo or intrinsics file found in repo; offline diagnostics used default intrinsics `fx=fy=570.3, cx=319.5, cy=239.5` (common pipeline defaults).

## What I verified (confirmed observations)

1. Topic subscriptions and code behavior
   - Both `grasp_estimator.py` and `depth_to_pointcloud.py` subscribe to `/camera_info`. The expected CameraInfo topic name is `/camera_info` in the running pipeline.
   - `grasp_estimator.py` backprojects masked pixels using the direct pinhole formula with the depth image values as-is; it does not apply the range→Z correction (`ray_scale`) present in `depth_to_pointcloud.py`.
   - `depth_to_pointcloud.py` converts range-to-Z and then applies a hard-coded world transform before publishing a `PointCloud2`. That path is distinct from `grasp_estimator.py`'s masked-pixel backprojection.

2. Offline reproduction used the same backprojection formula as `grasp_estimator.py` and the same depth image stored in `depth_maps/`; using those inputs I reproduced the raw point cloud geometry reported in `POINT_CLOUD_AUDIT.md` (X extent ≈ 0.077 m, N points = 2982, centroid and min/max as recorded).

3. Depth image used offline (`depth_maps/world1_primitives_depth.npy`) matches expected resolution and data type (saved as float32); boundary pixels in the mask have valid depth values.

## Remaining uncertainties (could not verify from repository alone)

- Live CameraInfo values (exact `fx, fy, cx, cy`, and image `width`/`height`) actually received at runtime by the nodes: there is no persisted CameraInfo message or intrinsics file in the workspace to confirm these values. Without the live `CameraInfo` we cannot prove the runtime intrinsics match the defaults used offline.

- Any transient runtime preprocessing applied by `depth_publisher.py` or by the ROS 2 environment (e.g., image rescaling, reprojection, or plugin-level depth post-processing) prior to the depth image being published to `/depth_model/depth_image` is not recorded in the saved `.npy` depth map, so we cannot confirm the offline depth image is byte-for-byte identical to the one consumed live at the same timestamp.

- Live timestamps and exact frame correlation: the offline depth file lacks the original ROS header timestamps and any camera frame metadata that would prove the same physical frame was used at runtime for a particular `grasp_estimator` evaluation tick.

## Runtime reproduction step (what I executed offline)

- Used `depth_maps/world1_primitives_depth.npy` and `data/masks/world1_primitives/obj_sphere_mask.npy`.
- Applied the same pinhole backprojection math as `grasp_estimator.py`:
  - `x = (u - cx) * z / fx`
  - `y = (v - cy) * z / fy`
  - `z = z` (depth value used as provided)
- Used defaults `fx=fy=570.3, cx=319.5, cy=239.5` (no saved CameraInfo available).
- Resulting raw cloud statistics (from `POINT_CLOUD_AUDIT.md`) matched the earlier offline audit: X extent ≈ 0.07707 m; N = 2982; centroid ≈ [-0.10222822, 0.25424197, 0.67350521].

## Conclusion — does the offline audit faithfully reproduce the live estimator?

Partially. The offline audit reproduces the same backprojection code path that `grasp_estimator.py` executes and reproduces the raw point-cloud geometry when using the saved depth image and the assumed intrinsics. However, because the repository lacks the actual `CameraInfo` messages and there is no record of runtime preprocessing steps or per-frame headers, I cannot fully confirm byte-for-byte equivalence with the live runtime inputs.

Key unresolved items that prevent a definitive yes:
- Exact `CameraInfo` values used by the live nodes (fx, fy, cx, cy, image w/h) were not accessible from the repository; if they differ from the defaults used offline, reprojection scale would change.
- Any runtime depth-image preprocessing (scaling, smoothing, encoding differences, or different depth-source semantics such as range vs Z) between what's saved in `depth_maps/` and what the estimator actually consumed at runtime cannot be ruled out from the saved artifacts alone.

If you want a definitive runtime-level check, the next data collection step (non-invasive) is to capture the live `/camera_info` message and a `rosbag` or exported headered depth frame matching the timestamp used by the estimator, then rerun the offline backprojection using those exact intrinsics and the recorded depth frame. I can help automate those capture commands if you have the live runtime environment available.

---

Files referenced: `scripts/grasp_estimator.py`, `scripts/depth_to_pointcloud.py`, `depth_maps/world1_primitives_depth.npy`, `data/masks/world1_primitives/obj_sphere_mask.npy`, `ground_truth/world1_primitives_pose.json`, `POINT_CLOUD_AUDIT.md`.
