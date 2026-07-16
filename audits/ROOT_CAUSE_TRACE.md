# Root Cause Trace Audit

## Pipeline Diagram

1. Gazebo depth camera publishes raw depth on `/depth_camera`.
2. `scripts/save_depth_gz.py` consumes `/depth_camera` and saves `depth_maps/world1_primitives_depth.npy`.
3. `ros_gz_bridge` publishes Gazebo RGB and camera info into ROS 2.
4. `scripts/depth_publisher.py` subscribes to `/camera/rgb/image_raw`, runs model inference, and publishes `/depth_model/depth_image`.
5. `scripts/grasp_estimator.py` subscribes to `/depth_model/depth_image` and `/camera_info`.
6. `grasp_estimator.py` applies a semantic mask, back-projects masked pixels, and calls `_estimate_dimensions()`.

## Stage-by-Stage Comparison

| Stage | Data Source | Resolution | Encoding | Dtype | Semantics | Notes |
|---|---|---|---|---|---|---|
| Offline saved map | `depth_maps/world1_primitives_depth.npy` from `scripts/save_depth_gz.py` | same as `/depth_camera` from Gazebo | raw float32 buffer | float32 | raw Gazebo depth values from `/depth_camera` | Saved directly from `/depth_camera` without conversion. |
| `/depth_camera` | Gazebo depth camera topic | sensor native | float32 | float32 | Gazebo depth output (likely ray/range depth) | `save_depth_gz.py` reads and saves raw bytes. |
| `/camera/rgb/image_raw` | Gazebo RGB camera | likely `rgb8`/`bgr8` | uint8 | uint8 | color image | input to depth model in `depth_publisher.py`. |
| `/depth_model/depth_image` | `scripts/depth_publisher.py` | model output resolution | `32FC1` | float32 | learned metric depth estimate (model output) | Produced by model inference, not forwarded from `/depth_camera`. |
| `grasp_estimator.py` internal | `self.latest_depth_image` from `/depth_model/depth_image` | whatever model publishes | `32FC1` | float32 | assumed metric Z-depth | directly used in pinhole backprojection. |

## Exact First Point of Divergence

- File: `scripts/depth_publisher.py`
- Function: `DepthPublisher._rgb_callback`
- Line numbers: 180-193
- Variables: `bgr`, `depth`

### Code responsible

```python
depth = infer_image_metric(
    self.model, bgr, self.input_size, self.device
)
...
depth_msg = self.bridge.cv2_to_imgmsg(
    depth.astype(np.float32), encoding="32FC1"
)
```

### Why this is the divergence point

- The offline depth file `depth_maps/world1_primitives_depth.npy` is recorded from the Gazebo topic `/depth_camera`.
- The runtime node `scripts/depth_publisher.py` does not subscribe to `/depth_camera`.
- Instead, it generates a new depth image from RGB using `self.model.infer_image(...)` and publishes that to `/depth_model/depth_image`.
- Therefore the live depth image consumed by `grasp_estimator.py` is a learned model output, not the same data as the saved offline `.npy` map.

## Code Flow and Semantics

### `scripts/save_depth_gz.py`
- Subscribes to `/depth_camera`.
- Reads `msg.data` as `np.float32`.
- Saves raw `depth_maps/{scene_name}_depth.npy`.
- No resizing, filtering, clipping, or conversion is applied.

### `scripts/depth_publisher.py`
- Subscribes to `/camera/rgb/image_raw`.
- Converts the ROS image to OpenCV BGR via `CvBridge.imgmsg_to_cv2(..., desired_encoding="bgr8")`.
- Calls `infer_image_metric(self.model, bgr, self.input_size, self.device)`.
- Publishes the result as `32FC1` float32 depth on `/depth_model/depth_image`.
- The model inference step is the first place where the runtime data diverges from the offline saved depth source.

### `scripts/grasp_estimator.py`
- Subscribes to `/depth_model/depth_image` and `/camera_info`.
- Converts the message to `self.latest_depth_image` via `CvBridge.imgmsg_to_cv2(..., desired_encoding="32FC1")`.
- Applies semantic mask and keeps finite depths.
- Back-projects using direct pinhole formulas and passes values into `_estimate_dimensions()`.
- No additional resizing, interpolation, or depth conversion is applied here.

## Comparison of the Three Datasets

1. `depth_maps/world1_primitives_depth.npy`
   - Source: `scripts/save_depth_gz.py` reading `/depth_camera`.
   - Content: raw Gazebo depth frame saved as float32.

2. `/depth_camera`
   - Source: Gazebo depth camera topic.
   - Content: raw depth from the simulator, recorded by `save_depth_gz.py`.

3. `/depth_model/depth_image`
   - Source: `scripts/depth_publisher.py` model inference from `/camera/rgb/image_raw`.
   - Content: learned model depth output published as `32FC1`.

## Explanation of Dimension Change

- `grasp_estimator.py` uses the depth image values directly in backprojection.
- Any difference between the saved offline map and live `/depth_model/depth_image` scales the reconstructed 3D points.
- Because object width/height are computed from the masked backprojected point cloud, a different depth map changes the estimated object dimensions.
- The earliest divergence is at the runtime source node, so the mismatch is introduced before `grasp_estimator.py` ever sees the data.

## Conclusion

The single earliest line of code where the runtime data ceases to match the offline depth map is in `scripts/depth_publisher.py` at the model inference and publish step (`depth = infer_image_metric(...)` and `cv2_to_imgmsg(...)`). This node creates a new depth image from RGB rather than using the Gazebo `/depth_camera` depth data that produced `depth_maps/world1_primitives_depth.npy`.
