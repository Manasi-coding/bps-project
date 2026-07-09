# Live Runtime Audit — `obj_sphere`

## Objective

Collect live runtime evidence to determine whether the reconstructed sphere width (~0.077 m offline) matches the live pipeline, and identify where any divergence first appears.

> Note: I cannot access your live ROS 2 runtime from this environment. The file below contains exact commands and small helper scripts you can run in the live environment to capture the required data and reproduce the estimator's backprojection. Run them and paste the outputs back here (or I can parse them if you provide the saved files).

---

## 1) Capture CameraInfo and depth frame (run on the live machine)

Save as `save_live_camera_depth.py` and run while the pipeline is running:

```python
# save_live_camera_depth.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import json
import sys

class Saver(Node):
    def __init__(self):
        super().__init__('live_saver')
        self.bridge = CvBridge()
        self.cam = None
        self.depth = None
        self.create_subscription(CameraInfo, '/camera_info', self.cam_cb, 10)
        self.create_subscription(Image, '/depth_model/depth_image', self.depth_cb, 10)
        self.timeout_sec = 10.0

    def cam_cb(self, msg: CameraInfo):
        if self.cam is not None:
            return
        cam = {
            'width': msg.width,
            'height': msg.height,
            'distortion_model': msg.distortion_model,
            'D': list(msg.d),
            'K': list(msg.k),
            'P': list(msg.p),
        }
        with open('live_camera_info.json', 'w') as f:
            json.dump(cam, f, indent=2)
        self.cam = cam
        self.get_logger().info('Saved live_camera_info.json')
        self._maybe_exit()

    def depth_cb(self, msg: Image):
        if self.depth is not None:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
            return
        np.save('live_depth.npy', depth)
        # also save header info
        hdr = {'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec}, 'frame_id': msg.header.frame_id}
        with open('live_depth_header.json', 'w') as f:
            json.dump(hdr, f)
        self.depth = True
        self.get_logger().info('Saved live_depth.npy and live_depth_header.json')
        self._maybe_exit()

    def _maybe_exit(self):
        if self.cam is not None and self.depth is not None:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = Saver()
    try:
        rclpy.spin(node, timeout_sec=node.timeout_sec)
    except Exception:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run:

```bash
python3 save_live_camera_depth.py
```

This produces:
- `live_camera_info.json` — contains width/height, `K`, `D`, and distortion model
- `live_depth.npy` — raw depth array (32FC1 expected)
- `live_depth_header.json` — depth frame header (stamp, frame_id)

---

## 2) Reproduce the estimator's backprojection exactly (run on live machine)

Save as `reproject_live.py` and run in the same folder where `live_camera_info.json` and `live_depth.npy` were saved.

```python
# reproject_live.py
import numpy as np, json, os

cam = json.load(open('live_camera_info.json'))
depth = np.load('live_depth.npy')

width = cam['width']; height = cam['height']
K = np.array(cam['K']).reshape(3,3)
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

# Prepare u,v grid
h,w = depth.shape
assert (h==height and w==width), f'resolution mismatch: depth {depth.shape} vs CameraInfo {height}x{width}'

u_coords = np.arange(w)
v_coords = np.arange(h)
uu, vv = np.meshgrid(u_coords, v_coords)

z = depth.flatten()
u = uu.flatten(); v = vv.flatten()
# matches grasp_estimator: filter finite and 0.05 < z < 5.0
valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
z = z[valid]; u = u[valid]; v = v[valid]

# backproject used in grasp_estimator.py: x=(u-cx)*z/fx, y=(v-cy)*z/fy, z=z
xs = (u - cx) * z / fx
ys = (v - cy) * z / fy
zs = z
pts = np.stack([xs, ys, zs], axis=1)

# compute stats
n = len(pts)
min_xyz = pts.min(axis=0)
max_xyz = pts.max(axis=0)
extents = max_xyz - min_xyz
centroid = pts.mean(axis=0)

print('camera intrinsics:', fx,fy,cx,cy)
print('resolution', width, height)
print('dtype', depth.dtype)
print('depth min/max', np.nanmin(depth), np.nanmax(depth))
print('n_points', n)
print('min_xyz', min_xyz)
print('max_xyz', max_xyz)
print('extents', extents)
print('centroid', centroid)

# Save results
np.save('live_reprojection_points.npy', pts)
with open('live_reprojection_stats.json', 'w') as f:
    import json
    json.dump({'n_points': int(n), 'min_xyz': min_xyz.tolist(), 'max_xyz': max_xyz.tolist(), 'extents': extents.tolist(), 'centroid': centroid.tolist()}, f, indent=2)
```

Run:

```bash
python3 reproject_live.py
```

This will create `live_reprojection_stats.json` and `live_reprojection_points.npy`.

---

## 3) Compare with offline audit

After running the above, inspect `live_reprojection_stats.json` and compare these fields to the offline audit (from `POINT_CLOUD_AUDIT.md`):

- Number of valid points (offline: 2982)
- X/Y/Z min/max (offline X min/max: -0.14444975, -0.06738059)
- X/Y/Z extents (offline X extent: 0.07706916)
- Centroid (offline: [-0.10222822, 0.25424197, 0.67350521])

If you prefer I can parse the created JSON files if you upload them here.

---

## 4) What I can (and cannot) collect from this environment

- I cannot subscribe to live ROS topics or run the above capture scripts from this environment, so I cannot produce the requested live CameraInfo, depth frame, or reprojection stats myself.
- The scripts above are self-contained and non-invasive; running them will not modify your pipeline. They merely subscribe and save one sample of each required message.

---

## 5) Short checklist — minimal outputs to paste back here

- `live_camera_info.json` (or paste the `K` entries and width/height)
- `live_depth.npy` (or a small summary: shape, dtype, min, max)
- `live_reprojection_stats.json` (the reprojection numeric results)

If you run the capture scripts and paste the three JSON outputs (or attach the files), I will incorporate them into `LIVE_RUNTIME_AUDIT.md` and state whether the live reconstruction reproduces the ~0.077 m width and exactly where any divergence first appears.
