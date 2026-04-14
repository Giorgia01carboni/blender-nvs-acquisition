import objaverse
import os
import shutil
import trimesh

TARGET_DIR = "/home/xrdev/Desktop/GPU-tests/gaussian-splatting/GsplatTest/src/synthetic-acquisition-blender/raw_meshes"
os.makedirs(TARGET_DIR, exist_ok=True)

# Thresholds for mesh validation
MIN_VERTICES = 500        # too few = placeholder or icon, not a real human mesh
MAX_VERTICES = 500_000    # too many = unreasonably heavy
MIN_FACES = 200
MAX_BBOX_RATIO = 20.0     # max ratio between longest and shortest bbox axis (rejects flat/stretched)
MIN_BBOX_FILL = 0.05      # minimum ratio of mesh volume to bbox volume (rejects broken rigs with splayed limbs)


def is_valid_mesh(filepath: str) -> bool:
    """Loads the mesh with trimesh and rejects degenerate geometry."""
    try:
        scene = trimesh.load(filepath, force='scene')
    except Exception as e:
        print(f"  SKIP: failed to load ({e})")
        return False

    # Flatten scene to a single mesh for inspection
    if isinstance(scene, trimesh.Scene):
        if len(scene.geometry) == 0:
            print("  SKIP: empty scene (no geometry)")
            return False
        mesh = scene.dump(concatenate=True)
    else:
        mesh = scene

    verts = len(mesh.vertices)
    faces = len(mesh.faces)
    bbox = mesh.bounding_box.extents  # (dx, dy, dz)
    min_extent = min(bbox)
    max_extent = max(bbox)

    print(f"  Vertices: {verts}, Faces: {faces}, BBox: {bbox}")

    if verts < MIN_VERTICES:
        print(f"  SKIP: too few vertices ({verts})")
        return False
    if verts > MAX_VERTICES:
        print(f"  SKIP: too many vertices ({verts})")
        return False
    if faces < MIN_FACES:
        print(f"  SKIP: too few faces ({faces})")
        return False
    if min_extent == 0:
        print("  SKIP: degenerate bounding box (zero extent)")
        return False
    if max_extent / min_extent > MAX_BBOX_RATIO:
        print(f"  SKIP: extreme aspect ratio ({max_extent / min_extent:.1f}x)")
        return False

    # Reject broken rigs: if the mesh is watertight we use its volume,
    # otherwise fall back to convex hull volume as an approximation
    bbox_vol = bbox[0] * bbox[1] * bbox[2]
    if bbox_vol > 0:
        if mesh.is_watertight:
            mesh_vol = abs(mesh.volume)
        else:
            mesh_vol = mesh.convex_hull.volume
        fill_ratio = mesh_vol / bbox_vol
        print(f"  BBox fill ratio: {fill_ratio:.3f}")
        if fill_ratio < MIN_BBOX_FILL:
            print(f"  SKIP: mesh barely fills its bounding box ({fill_ratio:.3f}) — likely broken rig")
            return False

    return True


def fetch_humanoid_base(limit=6):
    print("Loading Objaverse LVIS annotations...")
    lvis_annotations = objaverse.load_lvis_annotations()

    person_uids = lvis_annotations.get('person', [])
    print(f"Identified {len(person_uids)} humanoid candidates.")

    if not person_uids:
        return {}

    final_objects = {}
    batch_size = 10  # download in small batches to avoid wasting bandwidth

    for start in range(0, len(person_uids), batch_size):
        if len(final_objects) >= limit:
            break

        batch_uids = person_uids[start:start + batch_size]
        print(f"\nDownloading batch {start // batch_size + 1} ({len(batch_uids)} objects)...")
        cached_objects = objaverse.load_objects(uids=batch_uids)

        for uid, cached_path in cached_objects.items():
            if len(final_objects) >= limit:
                break

            print(f"Validating {uid}...")
            if not is_valid_mesh(cached_path):
                continue

            _, ext = os.path.splitext(cached_path)
            target_path = os.path.join(TARGET_DIR, f"{uid}{ext}")
            shutil.copy2(cached_path, target_path)
            final_objects[uid] = target_path
            print(f"  OK -> {target_path}")

    return final_objects


if __name__ == "__main__":
    downloaded_files = fetch_humanoid_base(limit=6)
    print(f"\nTotal valid meshes copied: {len(downloaded_files)}")
    for uid, path in downloaded_files.items():
        print(f"  {uid}: {path}")