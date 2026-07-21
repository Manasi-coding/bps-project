#!/usr/bin/env python3
import subprocess
import time
import signal
import os

WORLD_DIR = "worlds"
WORLD_FILES = [
    "world1_baseline.sdf",
    "world2_dense_clutter.sdf",
    "world3_thin_objects.sdf",
    "world4_support_scene.sdf",
    "world5_occlusion_scene.sdf",
    "world6_dense_mixed.sdf",
]

# Every world now has 6 total camera viewpoints (1 original + 5 added in
# Part 1), all sharing this naming convention:
#   viewpoint 1        -> scene "{world_id}",        topics /rgb_camera, /depth_camera, /semantic_camera/labels_map
#   viewpoint N (N>=2)  -> scene "{world_id}__v{N}",  topics /rgb_camera_v{N}, /depth_camera_v{N}, /semantic_camera_v{N}/labels_map
NUM_VIEWPOINTS = 6

MAX_WAIT = 30           # seconds to wait for topics on first attempt
FALLBACK_MAX_WAIT = 30  # seconds to wait after falling back to GUI mode

results = {}


def viewpoint_topics(n):
    """Return (rgb_topic, depth_topic, semantic_topic) for viewpoint n."""
    if n == 1:
        return "/rgb_camera", "/depth_camera", "/semantic_camera/labels_map"
    return f"/rgb_camera_v{n}", f"/depth_camera_v{n}", f"/semantic_camera_v{n}/labels_map"


def viewpoint_scene(world_id, n):
    """Return the scene name for viewpoint n of a given world."""
    return world_id if n == 1 else f"{world_id}__v{n}"


def wait_for_topics(topics, timeout):
    elapsed = 0
    while elapsed < timeout:
        try:
            listed = subprocess.run(
                ["gz", "topic", "-l"],
                capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            listed = ""
        if all(t in listed for t in topics):
            return True
        time.sleep(0.5)
        elapsed += 0.5
    return False


def kill_gazebo():
    subprocess.run(["pkill", "-9", "-f", "gz sim"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "ruby"], stderr=subprocess.DEVNULL)
    time.sleep(2)


def launch_gazebo(world_path, server_only=True):
    flags = "-s -r" if server_only else "-r"
    return subprocess.Popen(
        ["bash", "-c", f"LIBGL_ALWAYS_SOFTWARE=1 gz sim {flags} {world_path}"],
        preexec_fn=os.setsid
    )


def kill_process_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


for world in WORLD_FILES:
    world_id = world.replace(".sdf", "")
    world_path = f"{WORLD_DIR}/{world}"
    print("\n" + "=" * 60)
    print(f"Processing {world_id} ({NUM_VIEWPOINTS} viewpoints)")
    print("=" * 60)

    kill_gazebo()

    # Topics for every viewpoint must all be live before we start capturing,
    # since all viewpoint sensors are published by the same Gazebo process.
    all_topics = []
    for n in range(1, NUM_VIEWPOINTS + 1):
        all_topics.extend(viewpoint_topics(n))

    # Attempt 1: server-only (-s -r)
    print("Launching Gazebo (server-only, -s -r)...")
    gazebo = launch_gazebo(world_path, server_only=True)
    print(f"Waiting for camera topics from all {NUM_VIEWPOINTS} viewpoints...")
    ready = wait_for_topics(all_topics, MAX_WAIT)

    if not ready:
        print(f"✗ Server-only mode: required topics not found after {MAX_WAIT}s")
        print("Falling back to GUI mode (-r, no -s)...")
        kill_process_group(gazebo)
        time.sleep(2)
        kill_gazebo()

        gazebo = launch_gazebo(world_path, server_only=False)
        ready = wait_for_topics(all_topics, FALLBACK_MAX_WAIT)

        if not ready:
            print(f"✗ {world_id}: required topics never came up in either mode — skipping")
            results[world_id] = "FAILED (no topics in either mode)"
            kill_process_group(gazebo)
            time.sleep(2)
            continue
        else:
            print(f"✓ {world_id}: topics came up in GUI mode")

    world_ok = True
    for n in range(1, NUM_VIEWPOINTS + 1):
        scene = viewpoint_scene(world_id, n)
        rgb_topic, depth_topic, sem_topic = viewpoint_topics(n)

        print(f"\n--- {world_id} viewpoint {n} -> scene '{scene}' ---")

        steps = [
            ("RGB", ["python3", "scripts/save_rgb_gz.py", scene, rgb_topic]),
            ("Depth", ["python3", "scripts/save_depth_gz.py", scene, depth_topic]),
            ("Panoptic masks", ["python3", "scripts/export_masks.py", scene, sem_topic]),
            ("Boundary masks", ["python3", "scripts/export_boundary_mask.py", scene]),
        ]
        viewpoint_ok = True
        for label, cmd in steps:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"✗ {scene}: {label} step failed ({e})")
                viewpoint_ok = False
                break

        results[scene] = "OK" if viewpoint_ok else "FAILED (capture error)"
        if not viewpoint_ok:
            world_ok = False

    if not world_ok:
        print(f"\n✗ {world_id}: one or more viewpoints failed (see per-viewpoint results below)")

    print("Closing Gazebo...")
    kill_process_group(gazebo)
    time.sleep(3)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for scene, status in results.items():
    print(f"  {scene}: {status}")

failed = [s for s, st in results.items() if st != "OK"]
if failed:
    print(f"\n{len(failed)} scene(s) need attention: {failed}")
else:
    print("\nAll rooms/viewpoints processed successfully.")
