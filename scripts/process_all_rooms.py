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

RGB_TOPIC = "/rgb_camera"
DEPTH_TOPIC = "/depth_camera"
PANOPTIC_TOPIC = "/semantic_camera/labels_map"
MAX_WAIT = 30           # seconds to wait for topics on first attempt
FALLBACK_MAX_WAIT = 30  # seconds to wait after falling back to GUI mode

results = {}

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
    scene = world.replace(".sdf", "")
    world_path = f"{WORLD_DIR}/{world}"
    print("\n" + "=" * 60)
    print(f"Processing {scene}")
    print("=" * 60)

    kill_gazebo()

    # Attempt 1: server-only (-s -r)
    print("Launching Gazebo (server-only, -s -r)...")
    gazebo = launch_gazebo(world_path, server_only=True)
    print("Waiting for camera topics (rgb, depth, semantic)...")
    ready = wait_for_topics([RGB_TOPIC, DEPTH_TOPIC, PANOPTIC_TOPIC], MAX_WAIT)

    if not ready:
        print(f"✗ Server-only mode: required topics not found after {MAX_WAIT}s")
        print("Falling back to GUI mode (-r, no -s)...")
        kill_process_group(gazebo)
        time.sleep(2)
        kill_gazebo()

        gazebo = launch_gazebo(world_path, server_only=False)
        ready = wait_for_topics([RGB_TOPIC, DEPTH_TOPIC, PANOPTIC_TOPIC], FALLBACK_MAX_WAIT)

        if not ready:
            print(f"✗ {scene}: required topics never came up in either mode — skipping")
            results[scene] = "FAILED (no topics in either mode)"
            kill_process_group(gazebo)
            time.sleep(2)
            continue
        else:
            print(f"✓ {scene}: topics came up in GUI mode")

    scene_ok = True
    steps = [
        ("RGB", ["python3", "scripts/save_rgb_gz.py", scene]),
        ("Depth", ["python3", "scripts/save_depth_gz.py", scene]),
        ("Panoptic masks", ["python3", "scripts/export_masks.py", scene]),
        ("Boundary masks", ["python3", "scripts/export_boundary_mask.py", scene]),
    ]
    for label, cmd in steps:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ {scene}: {label} step failed ({e})")
            scene_ok = False
            break

    results[scene] = "OK" if scene_ok else "FAILED (capture error)"

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
    print("\nAll rooms processed successfully.")