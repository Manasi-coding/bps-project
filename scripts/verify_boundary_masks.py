import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Scenes to verify
scenes = [
    "isolated_usb",
    "isolated_pen",
    "isolated_bottle",
    "touching",
    "occluded",
    "thin_objects"
]

# Output directory
output_dir = "results/mask_verification"
os.makedirs(output_dir, exist_ok=True)

stats = []

for scene in scenes:

    # -----------------------------
    # File paths
    # -----------------------------
    rgb_path = f"rgb_images/{scene}_rgb.png"
    depth_path = f"depth_maps/{scene}_depth.npy"
    mask_path = f"boundary_masks/{scene}_mask.png"

    # -----------------------------
    # Check files exist
    # -----------------------------
    missing = []

    if not os.path.exists(rgb_path):
        missing.append(rgb_path)

    if not os.path.exists(depth_path):
        missing.append(depth_path)

    if not os.path.exists(mask_path):
        missing.append(mask_path)

    if missing:
        print(f"\nSkipping {scene}")
        for file in missing:
            print(f"  Missing: {file}")
        continue

    # -----------------------------
    # Load data
    # -----------------------------
    rgb = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # -----------------------------
    # Verify dimensions
    # -----------------------------

    # Depth and mask MUST match
    if depth.shape != mask.shape:
        print(f"Depth and mask size mismatch for {scene}")
        continue

    # Resize RGB only for visualisation
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(
            rgb,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    # -----------------------------
    # Create overlay
    # -----------------------------
    overlay = rgb.copy()
    overlay[mask > 0] = (0, 255, 0)

    blended = cv2.addWeighted(
        rgb,
        0.7,
        overlay,
        0.3,
        0
    )

    # -----------------------------
    # Statistics
    # -----------------------------
    boundary_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    ratio = boundary_pixels / total_pixels

    stats.append({
        "scene": scene,
        "boundary_pixels": boundary_pixels,
        "ratio": ratio
    })

    print(
        f"{scene}: "
        f"{boundary_pixels} boundary pixels "
        f"({ratio:.3%})"
    )

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB")

    im = axes[1].imshow(depth, cmap="plasma")
    axes[1].set_title("Depth")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Boundary Mask")

    axes[3].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Boundary Overlay")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(scene)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        os.path.join(output_dir, f"{scene}_verification.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# -----------------------------
# Save statistics
# -----------------------------
stats_path = os.path.join(output_dir, "stats.txt")

with open(stats_path, "w") as f:

    f.write("Boundary Mask Verification\n")
    f.write("==========================\n\n")

    for s in stats:
        f.write(f"Scene: {s['scene']}\n")
        f.write(f"Boundary pixels : {s['boundary_pixels']}\n")
        f.write(f"Boundary ratio  : {s['ratio']:.3%}\n")
        f.write("Visual verdict  : __________\n")
        f.write("\n")

print("\nVerification complete.")
print(f"Results saved to: {output_dir}")