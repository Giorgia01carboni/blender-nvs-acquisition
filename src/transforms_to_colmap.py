"""
Converts a NeRF-style transforms.json (output of blender_worker.py) into COLMAP
BINARY format so that gsplat's COLMAP Parser can read it directly.

We write binary (.bin) instead of text (.txt) because pycolmap's text parser has
a Python-3 bug (np.array(map(float, ...)) yields a 0-d object array).

Writes:
  <output_dir>/sparse/0/cameras.bin
  <output_dir>/sparse/0/images.bin
  <output_dir>/sparse/0/points3D.bin   (empty - use random init in gsplat)

Usage:
  python transforms_to_colmap.py --dataset_dir /path/to/dataset --resolution 800

The dataset_dir must contain transforms.json and images/.
After running, point gsplat's --data_dir at dataset_dir.
"""

import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


# COLMAP camera model IDs (see colmap/src/base/camera_models.h)
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
}


def fov_to_focal(fov_rad: float, image_size: int) -> float:
    return (image_size / 2.0) / math.tan(fov_rad / 2.0)


def rotmat_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> COLMAP quaternion (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def nerf_c2w_to_colmap_w2c(c2w: np.ndarray) -> np.ndarray:
    """
    Blender camera: +X right, +Y up, -Z forward.
    COLMAP camera:  +X right, +Y down, +Z forward.
    Flip Y/Z basis in camera frame, then invert (COLMAP wants W2C).
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(c2w @ flip)


def write_cameras_bin(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Writes a single PINHOLE camera with id=1."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))                                  # num_cameras
        f.write(struct.pack("<I", 1))                                  # camera_id
        f.write(struct.pack("<i", CAMERA_MODEL_IDS["PINHOLE"]))        # model_id
        f.write(struct.pack("<Q", width))                              # width
        f.write(struct.pack("<Q", height))                             # height
        f.write(struct.pack("<dddd", fx, fy, cx, cy))                  # params: fx, fy, cx, cy


def write_images_bin(path: Path, frames: list):
    """Writes one entry per frame: quaternion, translation, camera_id, name, no 2D points."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))                        # num_reg_images

        for i, frame in enumerate(frames, start=1):
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            w2c = nerf_c2w_to_colmap_w2c(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat_to_quaternion(R)
            name = os.path.basename(frame["file_path"])

            f.write(struct.pack("<I", i))                              # image_id
            f.write(struct.pack("<dddd", q[0], q[1], q[2], q[3]))      # qw qx qy qz
            f.write(struct.pack("<ddd", t[0], t[1], t[2]))             # tx ty tz
            f.write(struct.pack("<I", 1))                              # camera_id
            f.write(name.encode("utf-8") + b"\x00")                    # null-terminated name
            f.write(struct.pack("<Q", 0))                              # num_points2D = 0


def write_points3D_bin(path: Path, num_points: int = 100, extent: float = 1.0):
    """
    Writes a small dummy point cloud so gsplat's COLMAP Parser can compute
    scene scale / principal axes. The actual gaussian init still comes from
    --init_type random.
    """
    rng = np.random.default_rng(42)
    points = rng.uniform(-extent, extent, size=(num_points, 3))
    colors = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", num_points))                         # num_points
        for i in range(num_points):
            f.write(struct.pack("<Q", i + 1))                          # point3D_id
            f.write(struct.pack("<ddd", *points[i]))                   # xyz
            f.write(struct.pack("<BBB", *colors[i]))                   # rgb
            f.write(struct.pack("<d", 0.0))                            # error
            f.write(struct.pack("<Q", 0))                              # track_length = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Directory containing transforms.json and images/")
    parser.add_argument("--resolution", type=int, default=800)
    args = parser.parse_args()

    transforms_path = args.dataset_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"No transforms.json in {args.dataset_dir}")

    with open(transforms_path) as f:
        data = json.load(f)

    fov_x = data["camera_angle_x"]
    frames = data["frames"]
    W = H = args.resolution
    fx = fy = fov_to_focal(fov_x, W)
    cx, cy = W / 2.0, H / 2.0

    sparse_dir = args.dataset_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_bin(sparse_dir / "cameras.bin", W, H, fx, fy, cx, cy)
    write_images_bin(sparse_dir / "images.bin", frames)
    write_points3D_bin(sparse_dir / "points3D.bin")

    print(f"Wrote COLMAP binary files to {sparse_dir}")
    print(f"  cameras.bin:  1 PINHOLE camera ({W}x{H}, fx={fx:.2f})")
    print(f"  images.bin:   {len(frames)} frames")
    print(f"  points3D.bin: 100 dummy points (use --init_type random in trainer)")


if __name__ == "__main__":
    main()
