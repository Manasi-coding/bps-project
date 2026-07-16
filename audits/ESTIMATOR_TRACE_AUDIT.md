# Estimator Trace Audit — `obj_sphere`

## Objective

Collect runtime trace output from the instrumented `grasp_estimator.py` to capture the intermediate variables used to compute `estimated_width`, `estimated_height`, and `estimated_depth` for `obj_sphere`.

## What I changed (non-invasive)

- Added `TRACE` logging in `scripts/grasp_estimator.py` (guarded to only emit when `object_name == 'obj_sphere'`):
  - Input counts, median, MAD, depth gate, percentiles, kept count
  - PCA covariance matrix, eigenvalues, eigenvectors
  - Projection min/max and 5th/95th percentiles
  - Full projected spans and percentile spans

No algorithmic logic or thresholds were modified; this only emits additional info to the node logger.

## How to collect the trace (live runtime)

1. Start the pipeline so that `grasp_estimator.py` subscribes to `/depth_model/depth_image` and `/camera_info`.
2. Run `grasp_estimator.py` with parameters setting `object_name` to `obj_sphere` and `world_name` to `world1_primitives` (or as you normally run the estimator).

Example (ROS 2 launch / run command depends on your setup). If running the script directly in a test environment, ensure the node gets the same parameters as in your pipeline.

3. Capture the node logs for a single evaluation tick. The node logs will include lines prefixed with `TRACE INPUT obj_sphere`, `TRACE PCA obj_sphere`, `TRACE PROJ obj_sphere`, and `TRACE DIM obj_sphere`.

You can capture logs to a file (example using `ros2 run` style logging redirection):

```bash
# Example: adjust for how you run the node
ros2 run <pkg> grasp_estimator.py --ros-args -p object_name:=obj_sphere -p world_name:=world1_primitives 2>&1 | tee grasp_estimator_trace.log
```

Wait for the estimator to evaluate once (it runs on a timer). Then stop and attach `grasp_estimator_trace.log` or paste the `TRACE` lines here.

## What to paste here

Copy the `TRACE`-prefixed lines from the node log and paste them into this file under the corresponding headings below, or attach the full log file.

### TRACE INPUT (paste here)


### TRACE PCA (paste here)


### TRACE PROJ (paste here)


### TRACE DIM (paste here)


## Offline comparison (to be filled after capture)

- Offline full PCA-projected width: ~0.07515702 m
- Offline 5th–95th percentile width: ~0.05535453 m
- Runtime estimated width (from node log):


## First point of divergence

(To be filled after comparing the runtime TRACE values against the offline numbers.)

## Confirmed observations

(To be filled after capture.)

---

After you paste or attach the captured TRACE logs, I will fill in the comparison and identify the first variable where the runtime diverges from the offline audit.